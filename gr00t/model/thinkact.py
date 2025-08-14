# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from .gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from .policy import BasePolicy

class GemmaVisionProcessor(nn.Module):
    """Processes images for Gemma 3's vision encoder."""
    def __init__(self, image_size=896):
        super().__init__()
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    
    def forward(self, images):
        # Resize and normalize images for Gemma 3's vision encoder
        processed_images = []
        for img in images:
            # Convert to PIL Image if not already
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img.cpu())
            
            # Resize and normalize
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                self.normalize,
            ])
            processed_images.append(transform(img))
        
        return torch.stack(processed_images).to(images.device)


class ReasoningModule(nn.Module):
    """
    Multimodal LLM for high-level reasoning and planning using Gemma 3.
    Takes visual observations and language instructions, outputs reasoning traces
    and compressed visual plan latents.
    """
    def __init__(self, model_name: str = "google/gemma-3-27b-it"):
        super().__init__()
        # Initialize Gemma 3 model with vision capabilities
        self.mllm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vision_processor = GemmaVisionProcessor()
        
        # Projection layer to compress reasoning to visual plan latent
        self.plan_projection = nn.Sequential(
            nn.Linear(self.mllm.config.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)  # Visual plan latent dimension
        )
        
    def forward(self, 
               pixel_values: torch.Tensor, 
               input_text: str) -> Tuple[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, C, H, W) raw images (will be processed)
            input_text: Natural language instruction
            
        Returns:
            reasoning_trace: Natural language reasoning trace
            visual_plan: (B, D) Visual plan latent vector
        """
        # Process images for Gemma 3's vision encoder
        processed_images = self.vision_processor(pixel_values)
        
        # Prepare text prompt
        prompt = f"<start_of_turn>user\n{input_text}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize text
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            return_attention_mask=True
        ).to(pixel_values.device)
        
        # Forward pass through Gemma 3
        outputs = self.mllm(
            **inputs,
            pixel_values=processed_images,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last hidden state for plan projection
        last_hidden = outputs.hidden_states[-1]  # (B, L, D)
        
        # Use mean pooling for plan representation
        plan_rep = last_hidden.mean(dim=1)  # (B, D)
        
        # Project to visual plan latent
        visual_plan = self.plan_projection(plan_rep)
        
        # Generate reasoning trace
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        reasoning_trace = self.tokenizer.batch_decode(
            pred_ids[:, inputs.input_ids.shape[1]:],  # Only get new tokens
            skip_special_tokens=True
        )
        
        return reasoning_trace[0], visual_plan


@dataclass
class ThinkActConfig(GR00T_N1_5_Config):
    """Configuration for ThinkAct model."""
    model_type = "thinkact"
    
    # Reasoning module config
    reasoning_model_name: str = "liuhaotian/llava-v1.5-7b"
    visual_plan_dim: int = 512
    
    # RL training config
    use_rl: bool = True
    goal_reward_scale: float = 1.0
    traj_reward_scale: float = 0.1


class ThinkActPolicy(GR00T_N1_5):
    """
    ThinkAct policy that combines a reasoning MLLM with the GR00T action model.
    Implements the slow-thinking/fast-acting paradigm.
    """
    config_class = ThinkActConfig
    
    def __init__(self, config: ThinkActConfig, local_model_path: str):
        super().__init__(config, local_model_path)
        
        # Initialize reasoning module
        self.reasoning = ReasoningModule(config.reasoning_model_name)
        
        # Freeze the base model initially
        for param in self.parameters():
            param.requires_grad = False
            
        # Unfreeze action head for fine-tuning
        for param in self.action_head.parameters():
            param.requires_grad = True
            
        # Plan buffer for asynchronous operation
        self.current_plan = None
        self.plan_steps_remaining = 0
        self.plan_update_interval = 10  # Update plan every N steps
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Check if we need to update the plan
        if self.training or self.plan_steps_remaining <= 0:
            # Get visual and language inputs
            pixel_values = inputs["pixel_values"]
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            # Generate new plan
            reasoning_trace, visual_plan = self.reasoning(
                pixel_values, input_ids, attention_mask
            )
            
            # Store plan for multiple steps
            self.current_plan = visual_plan
            self.plan_steps_remaining = self.plan_update_interval
            
            if self.training:
                # In training, we might want to store additional info for RL
                inputs["reasoning_trace"] = reasoning_trace
        
        # Decrement plan counter
        self.plan_steps_remaining -= 1
        
        # Add visual plan to inputs for action model
        inputs["visual_plan"] = self.current_plan
        
        # Forward through base model
        return super().forward(inputs)
    
    def compute_reward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute RL reward for the reasoning module."""
        if not self.training or not self.config.use_rl:
            return 0.0
            
        # Extract predictions and targets
        pred_actions = batch["action_pred"]
        target_actions = batch["actions"]
        
        # Goal completion reward (L2 distance to target)
        goal_reward = -F.mse_loss(pred_actions, target_actions, reduction="none").mean()
        
        # Trajectory consistency reward (encourage smooth plans)
        if pred_actions.dim() > 2:  # If we have action sequences
            traj_reward = -pred_actions.diff(dim=1).abs().mean()
        else:
            traj_reward = torch.tensor(0.0, device=pred_actions.device)
        
        # Combine rewards
        total_reward = (
            self.config.goal_reward_scale * goal_reward +
            self.config.traj_reward_scale * traj_reward
        )
        
        return total_reward
