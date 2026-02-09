"""Training script for the Semantic Preference Model (SPM).

This script trains an SPM model that aligns text and motion representations
using contrastive learning with optional diffusion noise augmentation and
a retrieval-based fine-tuning objective.
"""

import os
import random

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from datetime import datetime
from diffusers import DDPMScheduler
from tqdm import tqdm

from mld.config import parse_args_RM
from mld.data.get_data import get_dataset
from mld.utils.utils import set_seed

from GradGuidance.spm import SPM, process_T5_outputs
from GradGuidance.utils import lengths_to_mask
from eval_tmr import (
    calculate_retrieval_metrics,
    calculate_retrieval_metrics_small_batches,
)

# Default checkpoint directory (override as needed)
CKPT_DIR = "checkpoints/spm"


def build_timestep_distribution(num_timesteps=1000, active_range=501, device="cuda"):
    """Build a probability distribution for sampling diffusion timesteps.

    Samples uniformly from [0, active_range) and assigns zero probability
    to timesteps in [active_range, num_timesteps).

    Args:
        num_timesteps: Total number of diffusion timesteps.
        active_range: Number of low-noise timesteps to sample from.
        device: Device to place the distribution tensor on.

    Returns:
        Probability distribution tensor of shape [num_timesteps].
    """
    probs = torch.zeros(num_timesteps, device=device)
    probs[:active_range] = 1.0 / active_range
    return probs


def summarize_parameters(model):
    """Summarize model parameter counts and weight sums by module group.

    Args:
        model: The SPM model to summarize.

    Returns:
        A formatted string with parameter statistics.
    """
    clip_count, clip_sum = 0, 0.0
    spm_count, spm_sum = 0, 0.0

    for name, param in model.named_parameters():
        if "clip" in name:
            clip_count += param.numel()
            clip_sum += param.float().sum().item()
        else:
            spm_count += param.numel()
            spm_sum += param.float().sum().item()

    return (
        f"\nCLIP Params: {clip_count / 1e6:.4g}M | CLIP Sum: {clip_sum}"
        f"\nSPM  Params: {spm_count / 1e6:.4g}M | SPM  Sum: {spm_sum}\n"
    )


@torch.no_grad()
def evaluate(test_loader, model, epoch=0):
    """Run evaluation on the test set using text-motion retrieval metrics.

    Computes both Text-to-Motion (T2M) and Motion-to-Text (M2T) retrieval
    metrics at various recall thresholds.

    Args:
        test_loader: DataLoader for the test set.
        model: The SPM model to evaluate.
        epoch: Current training epoch (for logging).
    """
    test_result = [[], [], []]

    for batch in tqdm(test_loader, desc="Evaluating", total=len(test_loader)):
        feats_ref = batch["motion"].float().cuda()
        text, m_len = batch["text"], batch["length"]

        t_len, token_emb, _ = process_T5_outputs(text, model.clip)
        m_latent = model.encode_motion(feats_ref, m_len)[0].squeeze().cpu().numpy()
        t_latent = model.encode_text(token_emb, t_len)[0].squeeze().cpu().numpy()

        for j in range(len(text)):
            test_result[0].append(text[j])
            test_result[1].append(t_latent[j])
            test_result[2].append(m_latent[j])

    # Apply deterministic shuffle for reproducible evaluation
    shuffle_index = np.load(f"./scripts/{len(test_result[0])}test_shuffle_index.npy")
    for k in range(3):
        test_result[k] = [test_result[k][i] for i in shuffle_index]

    # Text-to-Motion retrieval
    print("================== T2M Results ====================")
    calculate_retrieval_metrics_small_batches(test_result, epoch=epoch)
    calculate_retrieval_metrics(test_result, epoch=epoch)

    # Motion-to-Text retrieval (swap text and motion latents)
    test_result[1], test_result[2] = test_result[2], test_result[1]
    print("================== M2T Results ====================")
    calculate_retrieval_metrics_small_batches(test_result, epoch=epoch)
    calculate_retrieval_metrics(test_result, epoch=epoch)


def compute_finetune_loss(model, text, feats_ref, m_len, timestep, temperature=0.1):
    """Compute the fine-tuning loss using retrieval-based KL divergence.

    Constructs contrastive score pairs between top-k retrieved and ground-truth
    text-motion matches, then minimizes KL divergence against a target
    distribution that encourages correct retrievals.

    Args:
        model: The SPM model.
        text: List of text descriptions.
        feats_ref: Motion feature tensor [B, T, D].
        m_len: Motion lengths.
        timestep: Diffusion timesteps tensor.
        temperature: Softmax temperature for score scaling.

    Returns:
        KL divergence loss scalar.
    """
    with torch.no_grad():
        t_len, token_emb, _ = process_T5_outputs(text, model.clip)

    m_latent = model.encode_motion(feats_ref, m_len, timestep=timestep)[0].squeeze()
    t_latent = model.encode_text(token_emb, t_len, timestep=None)[0].squeeze()
    m_latent = F.normalize(m_latent, p=2, dim=1)
    t_latent = F.normalize(t_latent, p=2, dim=1)

    bs = m_latent.shape[0]
    sim_matrix = torch.matmul(t_latent, m_latent.T)

    # Check if ground-truth is in top-k or has high similarity
    _, topk_indices = torch.topk(sim_matrix, k=10, dim=1)
    gt_indices = torch.arange(bs, device=sim_matrix.device).view(-1, 1)
    in_topk = (topk_indices == gt_indices).any(dim=1).float()
    high_sim = (sim_matrix > 0.8).any(dim=1).float()
    correct_mask = torch.max(in_topk, high_sim)

    # Scale similarity scores by temperature
    sim_scaled = sim_matrix / temperature

    # Build contrastive logits: [retrieved_score, ground_truth_score]
    s_retrieved = sim_scaled.gather(1, topk_indices)[:, 0].squeeze()
    s_ground_truth = sim_scaled.diag()
    logits = torch.stack([s_retrieved, s_ground_truth], dim=1)

    # Build target distribution
    target = torch.zeros(bs, 2, device=m_latent.device)
    target[correct_mask.bool(), :] = 0.5      # Already correct: uniform
    target[~correct_mask.bool(), 1] = 1.0     # Incorrect: prefer ground truth

    # KL divergence loss
    probs = F.softmax(logits / temperature, dim=1)
    kl_loss = F.kl_div(probs.log(), target, reduction="none")
    return kl_loss.sum(dim=1).mean()


def save_checkpoint(model, epoch, cfg, finetune, maxT, step_aware, thr):
    """Save model checkpoint (excluding frozen CLIP weights).

    Args:
        model: The SPM model.
        epoch: Current epoch number.
        cfg: Configuration object.
        finetune: Whether fine-tuning mode is active.
        maxT: Maximum diffusion timestep.
        step_aware: Step-awareness mode string.
        thr: Noise threshold.
    """
    new_state = OrderedDict()
    for k, v in model.state_dict().items():
        if "clip" not in k:
            new_state[k] = v

    # Build checkpoint filename
    if finetune:
        prefix = f"FT-[{cfg.finetune.split('.')[0]}]-_"
    else:
        prefix = "SPM_"

    noise_tag = "Pure" if thr > 1 else "Noisy"
    filename = f"{prefix}{noise_tag}_T{maxT}_{step_aware}_E{epoch}.pth"

    os.makedirs(CKPT_DIR, exist_ok=True)
    save_path = os.path.join(CKPT_DIR, filename)
    torch.save({"state_dict": new_state}, save_path)
    print(f"Checkpoint saved: {save_path}")


def main():
    """Main training entry point."""
    # Parse configuration
    cfg = parse_args_RM()
    set_seed(cfg.SEED_VALUE)
    print(cfg)

    # Prepare data
    dataset = get_dataset(cfg)
    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()

    # Determine dataset-specific feature dimensions
    dataset_feats = {"humanml": 263, "kit": 251}
    if cfg.DATASET.NAME not in dataset_feats:
        raise ValueError(f"Unsupported dataset: {cfg.DATASET.NAME}")
    nfeats = dataset_feats[cfg.DATASET.NAME]

    # Initialize model
    model = SPM(
        clip_path="deps/sentence-t5-large",
        temp=cfg.CLTemp,
        thr=cfg.CLThr,
        nfeats=nfeats,
        ckpt_path=cfg.finetune,
    )
    model.train()
    model.cuda()

    # Initialize diffusion noise scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False,
    )

    # Training hyperparameters
    finetune = cfg.finetune is not None
    lr = 1e-4
    step_aware = cfg.step_aware
    maxT = cfg.maxT
    thr = cfg.NoiseThr
    temperature = 0.1

    if finetune:
        print(f"\n================ Fine-tuning from {cfg.finetune} ================")
        ckpt_path = os.path.join(CKPT_DIR, cfg.finetune)
        pretrained_ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(pretrained_ckpt, strict=False)
        lr = 1e-5

    print(
        f"================ Config: step_aware={step_aware}, maxT={maxT}, "
        f"NoiseThr={thr}, finetune={finetune} ================\n"
    )

    optimizer = torch.optim.AdamW(lr=lr, params=model.parameters())
    timestep_probs = build_timestep_distribution(device="cuda")

    # Training loop
    for epoch in range(501):
        print("Model Summary:", summarize_parameters(model))
        evaluate(test_loader, model, epoch)

        total_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch}", total=len(train_loader)
        ):
            feats_ref = batch["motion"].float().cuda()
            text, m_len = batch["text"], batch["length"]

            # Sample diffusion timesteps
            timestep = torch.multinomial(
                timestep_probs, num_samples=feats_ref.shape[0], replacement=True
            ).long()

            # Optionally add diffusion noise to motion features
            if random.random() > thr:
                with torch.no_grad():
                    noise = torch.randn_like(feats_ref)
                    feats_ref = scheduler.add_noise(
                        original_samples=feats_ref.clone(),
                        noise=noise,
                        timesteps=timestep,
                    )

            # Compute contrastive (SPM) loss
            spm_loss = model(
                text=text,
                motion_feature=feats_ref,
                m_len=m_len,
                timestep=timestep if finetune else None,
                mode=step_aware,
            )

            # Compute fine-tuning loss if applicable
            ft_loss = torch.tensor(0.0, device="cuda")
            if finetune:
                ft_loss = compute_finetune_loss(
                    model, text, feats_ref, m_len, timestep, temperature
                )

            loss = 1e-2 * ft_loss + spm_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{time_str}] Epoch {epoch} | Avg Loss: {avg_loss:.4f}\n")

        save_checkpoint(model, epoch, cfg, finetune, maxT, step_aware, thr)


if __name__ == "__main__":
    main()
