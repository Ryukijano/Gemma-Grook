#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import deque

import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    SiglipVisionModel,
    AutoImageProcessor,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.nn.functional as F

from lerobot.common.policies.gemma_le.configuration_gemma_le import GemmaLeConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class DiffusionSinusoidalPosEmb2D(nn.Module):
    """2D sinusoidal positional embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb_x = x[..., 0][:, None] * emb[None, :]
        emb_y = x[..., 1][:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb_x), torch.cos(emb_x), torch.sin(emb_y), torch.cos(emb_y)), dim=-1)
        return emb


class ScaleDPBlock(nn.Module):
    """A single ScaleDP transformer block with FiLM-like conditioning."""
    def __init__(self, config, action_dim, vlm_embedding_dim):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.vlm_embedding_dim = vlm_embedding_dim

        self.film_gamma = nn.Parameter(torch.ones(action_dim))
        self.film_beta = nn.Parameter(torch.zeros(action_dim))

        self.vlm_proj = nn.Linear(vlm_embedding_dim, action_dim)
        self.action_proj = nn.Linear(action_dim, action_dim)

    def forward(self, noised_action, timestep, conditioning):
        # FiLM conditioning
        gamma = self.film_gamma * conditioning
        beta = self.film_beta * conditioning

        # VLM conditioning
        vlm_conditioning = self.vlm_proj(conditioning)

        # Combine conditioning
        conditioning = gamma * vlm_conditioning + beta

        # Predict noise
        pred_noise = self.action_proj(conditioning)
        return pred_noise


class ScaleDPTransformer(nn.Module):
    """Scalable Diffusion Transformer for action generation with VLM conditioning."""
    def __init__(self, config, action_dim, vlm_embedding_dim):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.vlm_embedding_dim = vlm_embedding_dim

        self.film_gamma = nn.Parameter(torch.ones(action_dim))
        self.film_beta = nn.Parameter(torch.zeros(action_dim))

        self.vlm_proj = nn.Linear(vlm_embedding_dim, action_dim)
        self.action_proj = nn.Linear(action_dim, action_dim)

    def forward(self, noised_action, timestep, conditioning):
        """Predict noise with per-timestep conditioning.

        Inputs:
            noised_action: (B, T, A)
            timestep: (B,) or (B, T)
            conditioning: (B, F) fused features
        Returns:
            pred_noise: (B, T, A)
        """
        batch_size = noised_action.shape[0]
        seq_len = noised_action.shape[1]

        # Expand conditioning across time if needed: (B, F) -> (B, T, F)
        if conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(1).expand(batch_size, seq_len, -1)

        # Project fused conditioning to action space per step -> (B, T, A)
        vlm_conditioning = self.vlm_proj(conditioning)

        # FiLM on projected features (broadcast gamma/beta over (B,T,A))
        gamma = self.film_gamma.to(vlm_conditioning.dtype)
        beta = self.film_beta.to(vlm_conditioning.dtype)
        conditioned = gamma * vlm_conditioning + beta

        # Predict noise per step -> (B, T, A)
        conditioned = conditioned.to(self.action_proj.weight.dtype)
        pred_noise = self.action_proj(conditioned)
        return pred_noise


class GemmaLeModel(nn.Module):
    """Core model for Gemma-Le policy, integrating VLM and ScaleDP action head."""
    def __init__(self, config: GemmaLeConfig):
        super().__init__()
        self.config = config
        # Load Gemma-3 text model and SigLIP vision encoder
        self.text_model = AutoModelForCausalLM.from_pretrained(
            config.text_model_id, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_id)
        self.vision_model = SiglipVisionModel.from_pretrained(
            config.vision_model_id, torch_dtype=torch.bfloat16
        )
        self.image_processor = AutoImageProcessor.from_pretrained(config.vision_model_id)

        if config.lora_rank > 0:
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )
            self.text_model = get_peft_model(self.text_model, lora_config)

        if config.action_feature is None or config.action_feature.shape is None:
            raise ValueError("Action feature is not defined in the policy config.")
        action_dim = config.action_feature.shape[0]
        fused_dim = config.conditioning_dim
        # Infer hidden sizes robustly
        text_hidden_dim = getattr(self.text_model.config, "hidden_size", None)
        if text_hidden_dim is None and hasattr(self.text_model, "get_input_embeddings"):
            text_hidden_dim = self.text_model.get_input_embeddings().embedding_dim
        vision_hidden_dim = getattr(self.vision_model.config, "hidden_size", None)
        if vision_hidden_dim is None and hasattr(self.vision_model.config, "vision_config"):
            vision_hidden_dim = getattr(self.vision_model.config.vision_config, "hidden_size", None)
        if text_hidden_dim is None or vision_hidden_dim is None:
            raise RuntimeError("Could not infer hidden sizes for text/vision backbones.")
        self.fuse = nn.Sequential(
            nn.Linear(text_hidden_dim + vision_hidden_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
        )
        self.action_head = ScaleDPTransformer(config, action_dim, fused_dim)

    def get_conditioning(self, images: Tensor, input_ids: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        """Extract and fuse SigLIP vision + Gemma-3 text features.

        Ensures vision inputs are resized/normalized to the SigLIP expected resolution.
        """
        try:
            # Prepare vision inputs for SigLIP (expects (B,3,384,384))
            # Convert BCHW -> list of HWC numpy for processor
            imgs_hwc = images.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            proc = self.image_processor(images=list(imgs_hwc), return_tensors="pt", do_rescale=False)
            pixel_values = proc["pixel_values"].to(images.device, dtype=torch.bfloat16)

            vout = self.vision_model(pixel_values=pixel_values)
            vfeat = vout.pooler_output if hasattr(vout, "pooler_output") else vout.last_hidden_state.mean(dim=1)

            # Text: last hidden mean
            t_inputs = {"input_ids": input_ids, "output_hidden_states": True, "return_dict": True}
            if attention_mask is not None:
                t_inputs["attention_mask"] = attention_mask
            tout = self.text_model(**t_inputs)
            tfeat = tout.hidden_states[-1].mean(dim=1)
            fused = torch.cat([vfeat, tfeat], dim=-1)
            # Ensure dtype matches Linear weights
            fused = fused.to(self.fuse[0].weight.dtype)
            return self.fuse(fused)
        except Exception as e:
            raise RuntimeError(f"Conditioning failed: {e}")

    def forward(
        self, images: Tensor, input_ids: Tensor, attention_mask: Optional[Tensor], noised_action: Tensor, timestep: Tensor
    ) -> Tensor:
        # Expect images as (B, N, C, H, W); use first view
        conditioning = self.get_conditioning(images[:, 0], input_ids, attention_mask)

        # Predict noise
        pred_noise = self.action_head(noised_action, timestep, conditioning)
        return pred_noise 


class GemmaLePolicy(PreTrainedPolicy):
    """Gemma-Le policy for training and inference in LeRobot."""

    config_class = GemmaLeConfig
    name = "gemma_le"  # type: ignore

    def __init__(
        self,
        config: GemmaLeConfig,
        dataset_stats: Optional[dict[str, dict[str, Tensor]]] = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_id)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
        )

        self.model = GemmaLeModel(config)
        self._queues = None
        self.reset()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
        }
        self.action_pred_queue = deque(maxlen=self.config.n_action_steps)
        self.action_dim = self.config.output_features["action"].shape[0]
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    def get_optim_params(self) -> list[dict]:
        params = list(self.model.parameters())
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        return [{'params': params, 'lr': self.config.optimizer_lr}]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action during inference, managing queue and generation."""
        try:
            # Normalize inputs and targets
            batch = self.normalize_inputs(batch)
            batch = self.normalize_targets(batch)

            # Prepare inputs for the model
            images = torch.stack([batch[key] for key in self.config.image_features], dim=1)
            # Tokenize task text if present; otherwise use empty prompts
            texts = batch.get("task", None)
            if texts is None:
                texts = [""] * images.shape[0]
            tok = self.tokenizer(texts, return_tensors="pt", padding=True).to(images.device)

            # Forward diffusion
            trajectory = batch["action"]
            # Sample noise to add to the trajectory
            noise = torch.randn_like(trajectory)
            # Sample a random noising timestep for each item in the batch
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(trajectory.shape[0],),
                device=trajectory.device,
            ).long()
            # Add noise to the clean trajectories according to the noise magnitude at each timestep
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

            # Predict noise
            pred_noise = self.model(
                images,
                tok.get("input_ids"),
                tok.get("attention_mask"),
                noisy_trajectory,
                timesteps,
            )

            # Compute loss
            loss = F.mse_loss(pred_noise, noise)

            return loss, {"loss": loss.item()}
        except ValueError as e:
            raise RuntimeError(f"Inference input failed: {e}")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Optional[dict]]:
        """Forward pass for training: normalize, diffuse, predict noise, compute loss."""
        try:
            # Normalize inputs and targets
            batch = self.normalize_inputs(batch)
            batch = self.normalize_targets(batch)

            # Prepare inputs for the model
            images = torch.stack([batch[key] for key in self.config.image_features], dim=1)
            # Tokenize task text if present; otherwise use empty prompts
            texts = batch.get("task", None)
            if texts is None:
                texts = [""] * images.shape[0]
            tok = self.tokenizer(texts, return_tensors="pt", padding=True).to(images.device)

            # Forward diffusion
            trajectory = batch["action"]
            # Sample noise to add to the trajectory
            noise = torch.randn_like(trajectory)
            # Sample a random noising timestep for each item in the batch
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(trajectory.shape[0],),
                device=trajectory.device,
            ).long()
            # Add noise to the clean trajectories according to the noise magnitude at each timestep
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

            # Predict noise
            pred_noise = self.model(
                images,
                tok.get("input_ids"),
                tok.get("attention_mask"),
                noisy_trajectory,
                timesteps,
            )

            # Compute loss
            loss = F.mse_loss(pred_noise, noise)

            return loss, {"loss": loss.item()}
        except ValueError as e:
            raise RuntimeError(f"Input preparation failed: {e}") 