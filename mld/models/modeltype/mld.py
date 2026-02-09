"""MLD (Motion Latent Diffusion) model implementation.

This module implements the MLD model for text-to-motion generation using
diffusion models with reward-guided fine-tuning support.
"""

import inspect
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from mld.data.base import BaseDataModule
from mld.config import instantiate_from_config
from mld.utils.temos_utils import lengths_to_mask, remove_padding
from mld.utils.utils import (
    count_parameters,
    get_guidance_scale_embedding,
    extract_into_tensor,
    control_loss_calculate,
)

from .base import BaseModel
from GradGuidance.spm import process_T5_outputs
from GradGuidance.finetune_config import ft_reset

logger = logging.getLogger(__name__)


class MLD(BaseModel):
    """Motion Latent Diffusion model for text-to-motion generation."""

    def __init__(self, cfg: DictConfig, datamodule: BaseDataModule) -> None:
        super().__init__()
        self.reward_model = None
        self.cfg = cfg
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.datamodule = datamodule

        if cfg.model.guidance_scale == "dynamic":
            s_cfg = cfg.model.scheduler
            self.guidance_scale = s_cfg.cfg_step_map[s_cfg.num_inference_steps]
            logger.info(f"Guidance Scale set as {self.guidance_scale}")

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        self.vae = instantiate_from_config(cfg.model.motion_vae)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

        self._get_t2m_evaluator(cfg)

        self.metric_list = cfg.METRIC.TYPE
        self.configure_metrics()

        self.feats2joints = datamodule.feats2joints

        self.vae_scale_factor = cfg.model.get("vae_scale_factor", 1.0)
        self.guidance_uncondp = cfg.model.get("guidance_uncondp", 0.0)

        logger.info(f"vae_scale_factor: {self.vae_scale_factor}")
        logger.info(f"prediction_type: {self.scheduler.config.prediction_type}")
        logger.info(f"guidance_scale: {self.guidance_scale}")
        logger.info(f"guidance_uncondp: {self.guidance_uncondp}")

        self.is_controlnet = False

        self.summarize_parameters()
        self.gflops = 0
        self.update_cnt = 0

    @property
    def do_classifier_free_guidance(self) -> bool:
        """Check if classifier-free guidance is enabled."""
        return self.guidance_scale > 1 and self.denoiser.time_cond_proj_dim is None

    def summarize_parameters(self) -> None:
        """Log model parameter counts."""
        logger.info(f"VAE Encoder: {count_parameters(self.vae.encoder)}M")
        logger.info(f"VAE Decoder: {count_parameters(self.vae.decoder)}M")
        logger.info(f"Denoiser: {count_parameters(self.denoiser)}M")

    def forward(self, batch: dict, add_noise=None) -> tuple:
        """Forward pass for inference.

        Args:
            batch: Input batch containing text, motion, length, mask.
            add_noise: Optional function to add noise to latents.

        Returns:
            Tuple of (joints_result, joints_reference) or feats_result if add_noise is provided.
        """
        texts = batch["text"]
        feats_ref = batch.get("motion")
        lengths = batch["length"]
        mask = batch["mask"]

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)

        t_len, token_embeddings, text_emb = process_T5_outputs(
            texts, self.text_encoder.text_model
        )
        controlnet_cond = None

        feats_ref = feats_ref.cuda()
        mask = lengths_to_mask(lengths, feats_ref.device)
        batch["mask"] = mask
        latents, _ = self.vae.encode(feats_ref, mask)

        if add_noise is not None:
            latents, timestep = add_noise(latents)
            timestep = torch.stack([timestep, timestep], dim=0).reshape(-1).cuda()
            latents = latents.cuda()
        else:
            timestep = None

        mask = batch.get("mask", lengths_to_mask(lengths, text_emb.device))

        latents = self._diffusion_reverse(
            latents, text_emb, controlnet_cond=controlnet_cond, diy_timestep=timestep, batch=batch
        )
        feats_rst = self.vae.decode(latents / self.vae_scale_factor, mask)

        if add_noise is not None:
            return feats_rst

        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if feats_ref is not None:
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref

    def predicted_origin(
        self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor
    ) -> tuple:
        """Predict the original sample from model output.

        Args:
            model_output: Model prediction.
            timesteps: Diffusion timesteps.
            sample: Noisy sample.

        Returns:
            Tuple of (predicted_original_sample, predicted_epsilon).
        """
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sigmas * model_output) / alphas
            pred_epsilon = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alphas * model_output) / sigmas
        else:
            raise ValueError(
                f"Invalid prediction_type {self.scheduler.config.prediction_type}."
            )

        return pred_original_sample, pred_epsilon

    def _diffusion_reverse(
        self,
        latents: torch.Tensor,
        cond: torch.Tensor,
        controlnet_cond: Optional[torch.Tensor] = None,
        token_embeddings=None,
        batch=None,
        diy_timestep=None,
    ) -> torch.Tensor:
        """Reverse diffusion process for sampling.

        Args:
            latents: Initial latents.
            cond: Conditioning embeddings.
            controlnet_cond: Optional controlnet conditioning.
            token_embeddings: Optional token embeddings.
            batch: Optional batch dictionary.
            diy_timestep: Optional custom timestep.

        Returns:
            Denoised latents.
        """
        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(cond.device)

        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        timestep_cond = None
        if self.denoiser.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                latents.shape[0]
            )
            timestep_cond = get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.denoiser.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        if self.is_controlnet and self.do_classifier_free_guidance:
            controlnet_cond = torch.cat([controlnet_cond] * 2)

        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            controlnet_residuals = None
            if diy_timestep is not None:
                with torch.no_grad():
                    model_output = self.denoiser(
                        sample=latent_model_input,
                        timestep=diy_timestep,
                        timestep_cond=None,
                        encoder_hidden_states=cond,
                        controlnet_residuals=None,
                    )[0]
            else:
                with torch.no_grad():
                    model_output = self.denoiser(
                        sample=latent_model_input,
                        timestep=t,
                        timestep_cond=None,
                        encoder_hidden_states=cond,
                        controlnet_residuals=None,
                    )[0]

            if self.do_classifier_free_guidance:
                model_output_text, model_output_uncond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (
                    model_output_text - model_output_uncond
                )

            x_i_0, _ = self.predicted_origin(
                model_output, torch.tensor([t]).long().cuda(), latents
            )
            motion_x0 = self.vae.decode(x_i_0 / self.vae_scale_factor, batch["mask"])
            latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample
            motion_xt = self.vae.decode(latents / self.vae_scale_factor, batch["mask"])

            if hasattr(self, "motion_list_x0"):
                self.motion_list_x0.append(
                    self.feats2joints(motion_x0).detach().cpu().numpy()
                )
                self.motion_list_xt.append(
                    self.feats2joints(motion_xt).detach().cpu().numpy()
                )

        return latents

    def _diffusion_process(
        self, latents: torch.Tensor, encoder_hidden_states: torch.Tensor
    ):
        """Forward diffusion process for training.

        Args:
            latents: Clean latents.
            encoder_hidden_states: Text embeddings.

        Returns:
            Dictionary containing noise prediction results.
        """
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)

        timestep_cond = None

        controlnet_residuals = None
        router_loss_controlnet = None

        model_output, router_loss = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            timestep_cond=timestep_cond,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_residuals=controlnet_residuals,
        )

        latents_pred, noise_pred = self.predicted_origin(
            model_output, timesteps, noisy_latents
        )

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "sample_pred": latents_pred,
            "sample_gt": latents,
            "router_loss": router_loss_controlnet if self.is_controlnet else router_loss,
        }
        return n_set

    def train_diffusion_forward(self, batch: dict) -> dict:
        """Training forward pass for diffusion model.

        Args:
            batch: Training batch.

        Returns:
            Dictionary containing loss components.
        """
        feats_ref = batch["motion"]
        mask = batch["mask"]
        hint = batch.get("hint", None)
        hint_mask = batch.get("hint_mask", None)

        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, mask)
            z = z * self.vae_scale_factor

        text = batch["text"]
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i for i in text
        ]
        text_emb = self.text_encoder(text)
        n_set = self._diffusion_process(z, text_emb)

        loss_dict = dict()

        if self.denoiser.time_cond_proj_dim is not None:
            pass
        else:
            if self.scheduler.config.prediction_type == "epsilon":
                model_pred, target = n_set["noise_pred"], n_set["noise"]
            elif self.scheduler.config.prediction_type == "sample":
                pass
            else:
                raise ValueError(
                    f"Invalid prediction_type {self.scheduler.config.prediction_type}."
                )

            reward = torch.tensor(0.0, device=model_pred.device)
            diff_loss = F.mse_loss(model_pred, target, reduction="mean")

        loss_dict["diff_loss"] = diff_loss
        loss_dict["reward"] = reward
        loss_dict["router_loss"] = (
            n_set["router_loss"]
            if n_set["router_loss"] is not None
            else torch.tensor(0.0, device=diff_loss.device)
        )

        if self.is_controlnet and self.vaeloss:
            feats_rst = self.vae.decode(
                n_set["sample_pred"] / self.vae_scale_factor, mask
            )

            if self.cond_ratio != 0:
                joints_rst = self.feats2joints(feats_rst)
                if self.use_3d:
                    hint = self.datamodule.denorm_spatial(hint)
                else:
                    joints_rst = self.datamodule.norm_spatial(joints_rst)
                hint_mask = hint_mask.sum(-1, keepdim=True) != 0
                cond_loss = control_loss_calculate(
                    self.vaeloss_type,
                    self.control_loss_func,
                    joints_rst,
                    hint,
                    hint_mask,
                )
                loss_dict["cond_loss"] = self.cond_ratio * cond_loss
            else:
                loss_dict["cond_loss"] = torch.tensor(0.0, device=diff_loss.device)

            if self.rot_ratio != 0:
                mask = mask.unsqueeze(-1)
                rot_loss = control_loss_calculate(
                    self.vaeloss_type, self.control_loss_func, feats_rst, feats_ref, mask
                )
                loss_dict["rot_loss"] = self.rot_ratio * rot_loss
            else:
                loss_dict["rot_loss"] = torch.tensor(0.0, device=diff_loss.device)
        else:
            loss_dict["cond_loss"] = loss_dict["rot_loss"] = torch.tensor(
                0.0, device=diff_loss.device
            )

        total_loss = sum(loss_dict.values())
        loss_dict["loss"] = total_loss
        return loss_dict

    def _diffusion_reverse_ft(self, x_i, cond, strategy):
        """Fine-tuning reverse diffusion process.

        Args:
            x_i: Initial latents.
            cond: Conditioning embeddings.
            strategy: Fine-tuning strategy.

        Returns:
            Denoised latents.
        """
        x_i = x_i * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(cond.device)

        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        for i, t in enumerate(timesteps):
            if "DRTune" == self.ft_config["type"]:
                latent_model_input = (
                    torch.cat([x_i.detach()] * 2)
                    if self.do_classifier_free_guidance
                    else x_i.detach()
                )
                latent_model_input = latent_model_input.detach()
            else:
                latent_model_input = (
                    torch.cat([x_i] * 2) if self.do_classifier_free_guidance else x_i
                )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            enable_grad = i in self.ft_config["enable_grad"]
            ctx = torch.enable_grad() if enable_grad else torch.no_grad()

            with ctx:
                if "DRTune" == self.ft_config["type"] or (
                    "DRaFT" == self.ft_config["type"]
                    and i not in self.ft_config["t_train"]
                ):
                    model_output = self.denoiser(
                        sample=latent_model_input, timestep=t, encoder_hidden_states=cond
                    )[0]
                    if i not in self.ft_config["t_train"]:
                        model_output = model_output.detach()
                else:
                    model_output = self.denoiser(
                        sample=latent_model_input, timestep=t, encoder_hidden_states=cond
                    )[0]

            if self.do_classifier_free_guidance:
                model_output_text, model_output_uncond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (
                    model_output_text - model_output_uncond
                )

            if (
                ("ReFL" == self.ft_config["type"] and i == self.ft_config["t_min"])
                or ("DRTune" == self.ft_config["type"] and i == self.ft_config["t_min"])
            ):
                batched_t = (
                    torch.tensor(t.item(), dtype=torch.long)
                    .unsqueeze(0)
                    .expand(x_i.shape[0])
                    .to(x_i.device)
                )
                x_i, _ = self.predicted_origin(model_output, batched_t, x_i)
                return x_i

            x_i = self.scheduler.step(model_output, t, x_i, **extra_step_kwargs).prev_sample

        return x_i

    def ft_diffusion_forward(self, batch: dict) -> dict:
        """Fine-tuning forward pass.

        Args:
            batch: Training batch.

        Returns:
            Dictionary containing loss components.
        """
        feats_ref = batch["motion"]
        mask = batch["mask"]

        text = batch["text"]
        if self.do_classifier_free_guidance:
            text = text + [""] * len(text)

        t_len, token_embeddings, text_emb = process_T5_outputs(
            text, self.text_encoder.text_model
        )
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, mask)
            z = z * self.vae_scale_factor

        x_T = torch.randn((feats_ref.shape[0], *self.latent_dim), device=text_emb.device)
        strategy = None
        x_0 = self._diffusion_reverse_ft(x_T, text_emb, strategy)
        self.ft_config = ft_reset(self.ft_config)

        loss_dict = dict()

        reward = torch.tensor(0.0, device=text_emb.device)
        if self.reward_model is not None and self.lambda_reward != 0:
            x_0 = x_0 / self.vae_scale_factor
            recons_motion = self.vae.decode(x_0, mask)
            t_len = t_len[: len(t_len) // 2]
            token_embeddings = token_embeddings[: token_embeddings.shape[0] // 2]
            reward = 0 - self.reward_model.get_reward_t2m(
                raw_texts=batch["text"],
                motion_feats=recons_motion,
                m_len=batch["length"],
                t_len=t_len,
                sent_emb=token_embeddings,
                timestep=torch.tensor(0, dtype=torch.long).to(x_0.device),
            )
            self.trn_reward.append(reward.item())

            reward = reward * self.lambda_reward

        diff_loss = F.mse_loss(x_0, z, reduction="mean")

        loss_dict["diff_loss"] = torch.tensor(0.0, device=text_emb.device)
        loss_dict["reward"] = reward

        total_loss = sum(loss_dict.values())
        loss_dict["loss"] = total_loss
        return loss_dict

    def t2m_eval(self, batch: dict) -> dict:
        """Text-to-motion evaluation.

        Args:
            batch: Evaluation batch.

        Returns:
            Dictionary containing evaluation results.
        """
        bs = len(batch["text"])

        mask = lengths_to_mask(lengths=batch["length"], device="cuda:0")
        batch["mask"] = mask
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]
        hint = batch.get("hint", None)
        hint_mask = batch.get("hint_mask", None)

        if self.datamodule.is_mm:
            texts = batch["text"]
            feats_ref = batch["motion"]
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0
            )
            mask = mask.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0
            )
            pos_ohot = pos_ohot.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0
            )
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0
            )
            hint = hint and hint.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0
            )
            hint_mask = hint_mask and hint_mask.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0
            )
        else:
            texts = batch["text"]
            feats_ref = batch["motion"]

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)

        t_len, token_embeddings, text_emb = process_T5_outputs(
            texts, self.text_encoder.text_model
        )
        batch["t_len"] = t_len[: len(t_len) // 2]
        token_embeddings = token_embeddings[: len(t_len) // 2]

        controlnet_cond = None

        latents = torch.randn((feats_ref.shape[0], *self.latent_dim), device=text_emb.device)

        self.motion_list_x0 = []
        self.motion_list_xt = []
        for i in range(4):
            latents = torch.randn(
                (feats_ref.shape[0], *self.latent_dim), device=text_emb.device
            )
            latents = self._diffusion_reverse(
                latents, text_emb, controlnet_cond=None, token_embeddings=token_embeddings, batch=batch
            )

        self.motion_list_x0 = np.array(self.motion_list_x0)
        self.motion_list_xt = np.array(self.motion_list_xt)

        feats_rst = self.vae.decode(latents / self.vae_scale_factor, mask)

        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(
            m_lens,
            eval(f"self.cfg.DATASET.{self.cfg.DATASET.NAME.upper()}.UNIT_LEN"),
            rounding_mode="floor",
        )

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }

        if "hint" in batch:
            hint_3d = self.datamodule.denorm_spatial(batch["hint"]) * batch["hint_mask"]
            rs_set["hint"] = hint_3d
            rs_set["hint_mask"] = batch["hint_mask"]

        return rs_set

    def allsplit_step(
        self,
        split: str,
        batch: dict,
        optimizer=None,
        lr_scheduler=None,
        policy_model=None,
    ) -> Optional[dict]:
        """Unified step function for different splits.

        Args:
            split: Split type ('train', 'val', 'test', 'finetune', 'finetune_EZTUne').
            batch: Input batch.
            optimizer: Optional optimizer.
            lr_scheduler: Optional learning rate scheduler.
            policy_model: Optional policy model for EZTune method.

        Returns:
            Loss dictionary or None.
        """
        if split in ["test", "val"]:
            rs_set = self.t2m_eval(batch)

            if self.datamodule.is_mm:
                metric_list = ["MMMetrics"]
            else:
                metric_list = self.metric_list

            for metric in metric_list:
                if metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "MMMetrics" and self.datamodule.is_mm:
                    getattr(self, metric).update(
                        rs_set["lat_rm"].unsqueeze(0), batch["length"]
                    )
                elif metric == "ControlMetrics":
                    getattr(self, metric).update(
                        rs_set["joints_rst"],
                        rs_set["hint"],
                        rs_set["hint_mask"],
                        batch["length"],
                    )
                else:
                    raise TypeError(f"Not support this metric: {metric}.")

        if split in ["train", "val"]:
            loss_dict = self.train_diffusion_forward(batch)
            return loss_dict
        elif split in ["finetune"]:
            loss_dict = self.ft_diffusion_forward(batch)
            return loss_dict
        elif split in ["finetune_EZTUne"]:
            loss_dict = self.ft_diffusion_forward_EZTUne(
                batch, optimizer, lr_scheduler, policy_model=policy_model
            )
            return loss_dict

    def ft_diffusion_forward_EZTUne(
        self, batch: dict, optimizer, lr_scheduler, policy_model=None
    ) -> dict:
        """EZTune fine-tuning forward pass.

        Args:
            batch: Training batch.
            optimizer: Optimizer.
            lr_scheduler: Learning rate scheduler.
            policy_model: Optional policy model.

        Returns:
            Dictionary containing loss components.
        """
        feats_ref = batch["motion"]
        mask = batch["mask"]
        text = batch["text"]
        if self.do_classifier_free_guidance:
            text = text + [""] * len(text)

        t_len, token_embeddings, text_emb = process_T5_outputs(
            text, self.text_encoder.text_model
        )

        x_T = torch.randn((feats_ref.shape[0], *self.latent_dim), device=text_emb.device)
        x_0, reward = self._diffusion_reverse_ft_EZTUne(
            x_T,
            text_emb,
            optimizer,
            batch,
            t_len[: len(t_len) // 2],
            token_embeddings[: len(t_len) // 2],
            lr_scheduler,
            policy_model=policy_model,
        )

        loss_dict = dict()
        loss_dict["diff_loss"] = torch.tensor(0.0, device=text_emb.device)
        loss_dict["reward"] = reward

        total_loss = sum(loss_dict.values())
        loss_dict["loss"] = total_loss
        return loss_dict

    def _diffusion_reverse_ft_EZTUne(
        self, x_i, cond, optimizer, batch, t_len, token_embeddings, lr_scheduler, policy_model=None
    ):
        """EZTune fine-tuning reverse diffusion process.

        Args:
            x_i: Initial latents.
            cond: Conditioning embeddings.
            optimizer: Optimizer.
            batch: Batch dictionary.
            t_len: Text lengths.
            token_embeddings: Token embeddings.
            lr_scheduler: Learning rate scheduler.
            policy_model: Optional policy model.

        Returns:
            Tuple of (final latents, reward).
        """
        x_i = x_i * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(cond.device)

        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([x_i] * 2) if self.do_classifier_free_guidance else x_i
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            enable_grad = i in self.ft_config["enable_grad"]
            ctx = torch.enable_grad() if enable_grad else torch.no_grad()

            with ctx:
                model_output = self.denoiser(
                    sample=latent_model_input, timestep=t, encoder_hidden_states=cond
                )[0]

            x_i_0, _ = self.predicted_origin(
                model_output, torch.tensor([t]).long().cuda(), latent_model_input
            )
            cond_x_i_0, uncond_x_i_0 = x_i_0.chunk(2)
            x_i_0 = uncond_x_i_0 + self.guidance_scale * (cond_x_i_0 - uncond_x_i_0)

            if self.do_classifier_free_guidance:
                model_output_text, model_output_uncond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (
                    model_output_text - model_output_uncond
                )

            x_i = self.scheduler.step(model_output, t, x_i, **extra_step_kwargs).prev_sample

            if not enable_grad:
                continue

            recons_motion = self.vae.decode(x_i_0, batch["mask"])
            reward = self.reward_model.get_reward_t2m(
                raw_texts=batch["text"],
                motion_feats=recons_motion,
                m_len=batch["length"],
                t_len=t_len,
                sent_emb=token_embeddings,
                timestep=torch.tensor(0, dtype=torch.long).to(x_i.device),
            )
            self.trn_reward.append(reward.item())

            reward = 0.0 - reward * self.lambda_reward

            diff_loss = torch.tensor(0).to(x_i.device)
            loss = diff_loss + reward

            if policy_model is None:
                loss.backward()
            else:
                loss.backward()
                for pA, pB in zip(
                    policy_model.denoiser.parameters(), self.denoiser.parameters()
                ):
                    pA.grad = pB.grad.clone()
                    pB.grad = None

            if policy_model is None:
                torch.nn.utils.clip_grad_norm_(
                    self.denoiser.parameters(), self.cfg.TRAIN.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    policy_model.denoiser.parameters(), self.cfg.TRAIN.max_grad_norm
                )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            del recons_motion
            torch.cuda.empty_cache()
            x_i = x_i.detach().requires_grad_()

        return x_i, reward
