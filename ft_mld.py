"""Fine-tuning script for MLD using reward-guided optimization.

This script fine-tunes a pre-trained MLD model using reward-guided optimization
with support for multiple fine-tuning methods: ReFL, DRaFT, DRTune, AlignProp, and EZtune.
"""

import os
import sys
import logging
import os.path as osp

from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
import swanlab
import diffusers
import transformers
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import print_table, set_seed, move_batch_to_device
from GradGuidance.spm import SPM
from GradGuidance.finetune_config import get_ft_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default checkpoint directory (override as needed)
SPM_CKPT_DIR = "checkpoints/spm"


def summarize_parameters(model):
    """Summarize model parameters grouped by module prefix.

    Args:
        model: The MLD model to summarize.

    Returns:
        Tuple of (parameter_counts_dict, parameter_sums_dict).
    """
    para_cnt, para_sum = {}, {}
    for name, para in model.named_parameters():
        module_prefix = name.split(".")[0]
        try:
            para_cnt[module_prefix] += para.sum().item()
            para_sum[module_prefix] += para.numel()
        except KeyError:
            para_cnt[module_prefix] = para.sum().item()
            para_sum[module_prefix] = para.numel()
    return para_cnt, para_sum


def save_checkpoint(model, epoch, cfg, metrics, cur_rp1, cur_fid):
    """Save model checkpoint with metadata.

    Args:
        model: The MLD model.
        epoch: Current epoch number.
        cfg: Configuration object.
        metrics: Validation metrics dictionary.
        cur_rp1: Current R@1 score.
        cur_fid: Current FID score.

    Returns:
        Path to saved checkpoint.
    """
    save_path = os.path.join(
        cfg.output_dir,
        "checkpoints",
        f"E{epoch}-R1-{round(cur_rp1, 3)}-FID-{round(cur_fid, 3)}.ckpt",
    )

    ckpt = dict(
        state_dict=model.state_dict(),
        ft_config=model.ft_config,
        metrics=metrics,
        reward_record=[model.reward_record, model.trn_reward],
    )

    model.on_save_checkpoint(ckpt)
    torch.save(ckpt, save_path)
    return save_path


@torch.no_grad()
def validation(target_model, val_dataloader, device, cfg, writer, global_step, ema=False):
    """Run validation and compute metrics.

    Args:
        target_model: The MLD model to validate.
        val_dataloader: Validation data loader.
        device: Device to run validation on.
        cfg: Configuration object.
        writer: TensorBoard/SwanLab writer.
        global_step: Current global training step.
        ema: Whether this is an EMA model.

    Returns:
        Tuple of (max_rp1, min_fid, metrics_dict).
    """
    target_model.denoiser.eval()
    val_loss_list = []
    for val_batch in tqdm(val_dataloader, desc="Validation"):
        val_batch = move_batch_to_device(val_batch, device)
        val_loss_dict = target_model.allsplit_step(split="val", batch=val_batch)
        val_loss_list.append(val_loss_dict)

    metrics = target_model.allsplit_epoch_end()
    metrics["Val/loss"] = sum([d["loss"] for d in val_loss_list]).item() / len(
        val_dataloader
    )
    metrics["Val/diff_loss"] = sum([d["diff_loss"] for d in val_loss_list]).item() / len(
        val_dataloader
    )
    metrics["Val/reward"] = sum([d["reward"] for d in val_loss_list]).item() / len(
        val_dataloader
    )

    max_val_rp1 = metrics["Metrics/R_precision_top_1"]
    min_val_fid = metrics["Metrics/FID"]
    print_table(f"Validation@Step-{global_step}", metrics)

    for mk, mv in metrics.items():
        mk = mk + "_EMA" if ema else mk
        if cfg.vis == "tb":
            writer.add_scalar(mk, mv, global_step=global_step)
        elif cfg.vis == "swanlab":
            writer.log({mk: mv}, step=global_step)

    target_model.denoiser.train()
    return max_val_rp1, min_val_fid, metrics


def main():
    """Main fine-tuning entry point."""
    cfg = parse_args()

    # Setup fine-tuning configuration
    ft_config = get_ft_config(
        ft_type=cfg.ft_type,
        m=cfg.ft_m,
        prob=cfg.ft_prob,
        t=cfg.ft_t,
        k=cfg.ft_k,
        skip=cfg.ft_skip,
        reverse=cfg.ft_reverse,
        custom=None,
        lambda_reward=cfg.ft_lambda_reward,
        dy=cfg.ft_dy,
    )
    spm_ckpt_path = os.path.join(SPM_CKPT_DIR, f"{cfg.spm_path}.pth")
    ft_config["reward_model_ckpt"] = spm_ckpt_path

    # Build experiment name
    ft_config["name"] += f"_lr{cfg.ft_lr:.0e}"
    if cfg.ft_dy != 2:
        ft_config["name"] += f"_dy{cfg.ft_dy}"

    cfg.ft_config = ft_config

    # Setup device and random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.SEED_VALUE)

    # Setup output directory
    cfg.output_dir = f"./checkpoints/ft_mld_step_iclr_re_final/{ft_config['name']}"
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=True)

    # Setup logging and visualization
    if cfg.vis == "tb":
        writer = SummaryWriter(cfg.output_dir)
    elif cfg.vis == "swanlab":
        writer = swanlab.init(
            project="MotionLCM",
            experiment_name=os.path.normpath(cfg.output_dir).replace(os.path.sep, "-"),
            suffix=None,
            config=dict(**cfg),
            logdir=cfg.output_dir,
        )
    else:
        raise ValueError(f"Invalid vis method: {cfg.vis}")

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(cfg.output_dir, "output.log"))
    handlers = [file_handler, stream_handler]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    OmegaConf.save(cfg, osp.join(cfg.output_dir, "config.yaml"))

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # Load dataset
    dataset = get_dataset(cfg)
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()

    # Initialize model
    model = MLD(cfg, dataset)

    assert cfg.TRAIN.PRETRAINED, "cfg.TRAIN.PRETRAINED must not be None."
    logger.info(f"Loading pre-trained model: {cfg.TRAIN.PRETRAINED}")
    state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
    logger.info(model.load_state_dict(state_dict, strict=False))

    # Freeze VAE and text encoder
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.vae.eval()
    model.text_encoder.eval()
    model.to(device)

    # Override learning rate from config
    cfg.TRAIN.learning_rate = float(cfg.ft_lr)
    logger.info(f"Learning rate: {cfg.TRAIN.learning_rate}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.denoiser.parameters(),
        lr=cfg.TRAIN.learning_rate,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon,
    )

    if cfg.TRAIN.max_ft_steps == -1:
        cfg.TRAIN.max_ft_steps = cfg.TRAIN.max_ft_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.lr_warmup_steps,
        num_training_steps=cfg.TRAIN.max_ft_steps,
    )

    # Training info
    logger.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logging.info(f"  Num Epochs = {cfg.TRAIN.max_ft_epochs}")
    logging.info(f"  Instantaneous batch size per device = {cfg.TRAIN.BATCH_SIZE}")
    logging.info(f"  Total optimization steps = {cfg.TRAIN.max_ft_steps}")
    logging.info(f"  1 Epoch == {len(train_dataloader)} Step")

    global_step = 0

    # Load reward model
    reward_model = SPM(ckpt_path=spm_ckpt_path, lambda_m2m=0, lambda_t2m=0).cuda()
    spm_state = torch.load(spm_ckpt_path, map_location="cpu")["state_dict"]
    reward_model.load_state_dict(spm_state, strict=False)
    model.reward_model = reward_model

    # Initialize fine-tuning config
    model.ft_config = ft_config
    model.lambda_reward = ft_config["lambda_reward"]
    model.reward_record = [[] for _ in range(50)]
    model.trn_reward = []

    logger.info(f"FineTune Config: {ft_config}")

    max_rp1, min_fid = 0.80862, 0.40636  # Baseline metrics
    rcd_reward = []
    device = torch.device("cuda:0")

    progress_bar = tqdm(range(0, cfg.TRAIN.max_ft_steps), desc="Steps")

    # Training loop
    for epoch in range(4):
        para_cnt, para_sum = summarize_parameters(model)
        logger.info(f"Epoch {epoch}: Para Sum: {para_sum} Para Cnt: {para_cnt}\n")

        for step, batch in enumerate(train_dataloader):
            batch = move_batch_to_device(batch, device)

            if ft_config["type"] == "EZtune":
                loss_dict = model.allsplit_step(
                    "finetune_EZtune", batch, optimizer, lr_scheduler
                )
            else:
                loss_dict = model.allsplit_step("finetune", batch)

            if ft_config["type"] != "EZtune":
                loss = loss_dict["loss"]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.denoiser.parameters(), cfg.TRAIN.max_grad_norm
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            diff_loss = loss_dict["diff_loss"]
            reward = loss_dict["reward"]
            rcd_reward.append(reward.item())
            loss = loss_dict["loss"]

            progress_bar.update(1)
            global_step += 1

            logs = {
                "Epoch": epoch,
                "loss": loss.item(),
                "diff_loss": 0,
                "lr": lr_scheduler.get_last_lr()[0],
                "reward": reward.item(),
            }
            progress_bar.set_postfix(**logs)

            for k, v in logs.items():
                if cfg.vis == "tb":
                    writer.add_scalar(f"ft/{k}", v, global_step=global_step)
                elif cfg.vis == "swanlab":
                    writer.log({f"ft/{k}": v}, step=global_step)

        # End of epoch: validation and checkpointing
        cur_rp1, cur_fid, metrics = validation(
            model, val_dataloader, device, cfg, writer, global_step
        )

        save_path = save_checkpoint(model, epoch, cfg, metrics, cur_rp1, cur_fid)

        main_metrics = {k: round(v, 4) for k, v in metrics.items() if "gt" not in k}
        logger.info(
            f"Epoch: {epoch}, Saved state to {save_path} with "
            f"R@1:{round(cur_rp1, 3)}, FID:{round(cur_fid, 3)}"
        )
        logger.info(f"Epoch: {epoch}, Main Metrics: {main_metrics}\n")


if __name__ == "__main__":
    main()
