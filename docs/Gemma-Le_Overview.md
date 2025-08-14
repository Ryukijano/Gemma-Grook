# Gemma-Le: A Practical Vision-Language-Action Policy for Robotic Manipulation

## Abstract

**Gemma-Le** integrates NVIDIA’s `GR00T-N1.5-3B` foundation policy with the modular **LeRobot** framework to create an end-to-end vision-language-action (VLA) system for robotic manipulation.  The model fuses a **SigLIP** vision encoder, a **Qwen3-1.7B** language backbone, and a diffusion-based action head, trained on the `robot_sim.PickNPlace` dataset.  We document the complete workflow—from dependency resolution and custom data configuration to multi-GPU training and evaluation—providing a reproducible recipe for researchers who wish to train large multimodal policies on commodity hardware.

---

## 1  Problem Statement & Real-World Impact

Robotic manipulation research faces four practical barriers:

1. **Complex Integration** – Cutting-edge components originate from different projects and often suffer dependency conflicts (e.g., PyTorch ⇄ CUDA ⇄ `flash-attn`).
2. **Hardware Accessibility** – Scaling large VLA models requires careful memory management across multiple GPUs to avoid frequent CUDA OOM errors.
3. **Data–Model Alignment** – Datasets rarely match model modality expectations out-of-the-box; custom configuration is required to resolve shape and normalization mismatches.
4. **Reproducibility** – Published results frequently omit the “hidden” engineering steps needed for a stable training run.

Gemma-Le directly addresses these issues by publishing every environment tweak, custom dataclass, and launch script required to reproduce our training pipeline.

---

## 2  Technical Architecture

### System Diagram
```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Multimodal    │    │  GR00T N1.5-3B   │    │     Action       │
│     Inputs      │───▶│    Backbone      │───▶│    Prediction     │
└─────────────────┘    └──────────────────┘    └──────────────────┘
│                      │                       │
├─ Ego-view Video      ├─ SigLIP Vision        ├─ Diffusion Policy
├─ Joint State         ├─ Qwen3 Language       ├─ 8-step Horizon
├─ EE Pose             ├─ EagleBackbone Fusion ├─ 7-DoF Joints
└─ Language Command    └─ Cross Attention      └─ Gripper State
```

| Component            | Params | Memory (BF16) | Trainable | Optimisation              |
|----------------------|--------|---------------|-----------|---------------------------|
| SigLIP Vision        | 86 M   | ≈ 172 MB      | ❌ Frozen  | Zero-shot transfer        |
| Qwen3-1.7B Language  | 1.7 B  | ≈ 3.4 GB      | ✅ Full    | Grad. checkpointing       |
| EagleBackbone Fusion | 92 M   | ≈ 184 MB      | ✅ Full    | Mixed precision           |
| Diffusion Head       | 11 M   | ≈ 44 MB       | ✅ Full    | Mixed precision           |
| **Total**            | **1.89 B** | **≈ 3.8 GB** | **≈ 1.8 B** | Multi-GPU distributed     |

---

## 3  Dataset & Data Configuration

* **LeRobot Dataset** – `robot_sim.PickNPlace`, thousands of simulated manipulation episodes with RGB video, proprioception, and action labels.
* **Custom Config** – `RobotSimPickNPlaceConfig` maps available modalities (only `video.ego_view`) to GR00T input expectations, resolving missing keys like `left_view` and `right_wrist_view`.

---

## 4  Training Methodology

```python
TRAINING_CONFIG = {
    "batch_size_per_device": 2,
    "gradient_accumulation_steps": 8,  # Effective BS = 48
    "precision": "bf16",
    "num_gpus": 3,
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine_with_restarts",
    "optimizer": "AdamW",
    "action_horizon": 8,
    "num_diffusion_timesteps": 100,
}
```

Key engineering steps:

1. **Dependency Resolution** – Installed a community‐built `flash-attn` wheel matching Python 3.10 + PyTorch 2.7.1 + CUDA 12.6; used `TRANSFORMERS_NO_FLASH_ATTENTION=1` as fallback.
2. **Memory Optimisation** – Enabled gradient checkpointing, used BF16, and tuned batch / accumulation to fit within 48 GB per L40S.
3. **Stable Scaling** – Established a working single-GPU baseline before moving to a 2-GPU DDP configuration (more stable than 3-GPU for NCCL).

---

## 5  Evaluation Pipeline

The `scripts/eval_thinkact.py` script outputs:

| Metric               | Description                                            |
|----------------------|--------------------------------------------------------|
| `action_mse`         | Mean-squared error between predicted and GT actions    |
| `success_rate`       | % episodes where `goal_reached == True` (if provided)  |
| `inference_time`     | Seconds per forward pass (CUDA events)                 |
| `samples_processed`  | Total number of eval samples                           |
| **Stat Aggregates**  | `avg_`, `std_`, `min_`, `max_` for each numeric metric |

All metrics are saved to `eval_results.json` for downstream analysis.

---

## 6  Reproducibility

* **Repository:** <https://github.com/Ryukijano/Gemma-Grook>
* **Model Weights:** <https://huggingface.co/Ryukijano/gemma-grook> (saved as `.safetensors`)
* **Artifacts Provided:**
  * `pyproject.toml` / environment YAML for exact dependency versions
  * Custom dataset config (`RobotSimPickNPlaceConfig`)
  * Launch scripts for single-GPU and multi-GPU runs (Accelerate)

Clone → `make setup` → `make train` → `make evaluate` reproduces our results end-to-end.

---

## 7  Key Lessons & Future Work

* **Environment Matters** – Binary compatibility (PyTorch × CUDA × flash-attn) is often the hidden blocker; document your exact environment.
* **Start Small** – Validate single-GPU training before scaling; most logic bugs surface early.
* **Custom Data Configs** – Even “standard” datasets may need modality remapping to meet model expectations.

Future improvements will explore **Think-Act reasoning modules** and light-weight **2-D RoPE** enhancements, but these are beyond the scope of Gemma-Le’s current release.

---

© 2025 Ryukijano — Released under Apache 2.0
