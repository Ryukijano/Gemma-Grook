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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TypeVar

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("gemma_le")
@dataclass
class GemmaLeConfig(PreTrainedConfig):
    """Configuration class for the Gemma-Le policy model."""

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture / modeling.
    # Backbones (explicit IDs)
    vision_model_id: str = field(
        default="google/siglip-so400m-patch14-384",
        metadata={"help": "Hugging Face ID for SigLIP vision encoder."},
    )
    text_model_id: str = field(
        default="google/gemma-3-4b-it",
        metadata={"help": "Hugging Face ID for Gemma 3 language model (decoder-only)."},
    )
    use_2d_rope: bool = field(
        default=False, metadata={"help": "Whether to use 2D Rotary Position Embeddings in the vision model."}
    )

    # LoRA config
    lora_rank: int = field(
        default=16,
        metadata={"help": "Rank for LoRA adapters in VLM fine-tuning."}
    )

    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "Attention projection module names to apply LoRA on in Gemma-3."},
    )

    # ScaleDP action head config
    scaledp_num_layers: int = 4
    scaledp_num_heads: int = 8
    scaledp_dim_model: int = 512
    scaledp_dim_feedforward: int = 2048

    # Diffusion config
    num_diffusion_steps: int = field(
        default=100,
        metadata={"help": "Number of diffusion steps for the denoising process."}
    )

    # Conditioning / planner
    conditioning_dim: int = field(
        default=768,
        metadata={"help": "Dimension of fused conditioning (vision+text or plan latent)."},
    )
    plan_update_interval: int = field(
        default=10,
        metadata={"help": "Steps between plan recomputation in ThinkAct-style usage."},
    )

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6



    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> Optional[LRSchedulerConfig]:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None 