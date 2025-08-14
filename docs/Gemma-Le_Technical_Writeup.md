# Gemma-Le: A Practical Guide to Training a 3B Vision-Language-Action Policy

---

## 1. Abstract

**Gemma-Le** represents a comprehensive, end-to-end implementation for training a large-scale Vision-Language-Action (VLA) model for robotic manipulation. Our project integrates NVIDIA's `GR00T-N1.5-3B` foundation policy—a powerful architecture fusing a **SigLIP** vision encoder and a **Qwen3-1.7B** language backbone—with the modular and accessible **LeRobot** framework. The primary contribution of this work is not a novel model architecture, but a transparent and reproducible documentation of the entire engineering journey. We address the critical, often-overlooked challenges of dependency management, multi-GPU training optimization, and data-to-model alignment. By providing a detailed account of overcoming common but significant hurdles like CUDA OOM errors, binary dependency conflicts (`flash-attn`), and dataset modality mismatches, we offer a practical blueprint for researchers and developers aiming to train sophisticated robotics models on commodity high-performance computing resources.

---

## 2. The Problem: Bridging the Gap Between Research and Reality

The field of AI-driven robotics is advancing at an unprecedented pace, yet a significant gap persists between state-of-the-art research and practical, reproducible implementation. We identified four primary barriers that hinder progress:

1.  **The Environment Morass (Dependency Hell):** Modern VLA models are complex compositions of components from different research ecosystems. Integrating them creates a fragile dependency web where specific versions of PyTorch, CUDA, and specialized libraries like `flash-attn` or `pytorch3d` must align perfectly. A single version mismatch can lead to cryptic binary incompatibility errors, halting progress for days.

2.  **The Multi-GPU Gauntlet:** Training billion-parameter models is computationally expensive and requires multi-GPU setups. However, scaling from a single GPU to a distributed environment introduces new failure modes, including CUDA Out-of-Memory (OOM) errors that are not simple overflows, network communication bottlenecks (NCCL), and silent process terminations (SIGTERM) that are difficult to debug.

3.  **The Data-Model Impedance Mismatch:** Foundation models are trained with specific expectations for input data—modalities, image resolutions, normalization statistics, and dictionary keys. Datasets, even when standardized, rarely match these expectations perfectly. Adapting a dataset to a model's input pipeline is a crucial but tedious process of configuration and debugging.

4.  **The Reproducibility Crisis:** Many academic papers and open-source projects present a clean, idealized version of their work. They often omit the crucial, messy details of the workarounds, environment tweaks, and failed attempts that were necessary to achieve a successful result. This makes it exceedingly difficult for others to reproduce their work and build upon it.

Our project tackles these challenges head-on by documenting our solutions to each of these problems in a real-world training scenario.

---

## 3. Technical Architecture: Deconstructing GR00T N1.5

Our system is built upon the `nvidia/GR00T-N1-5-3B` model, a pre-configured VLA policy designed for manipulation tasks. Its architecture is a powerful example of modern multimodal fusion.

### System Data Flow

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

*   **Vision Backbone (SigLIP):** The model uses a `SiglipVisionModel`. Unlike CLIP, which is trained on noisy web data, SigLIP (Signal-passing Language-Image Pre-training) is trained on a cleaner, more balanced dataset of image-text pairs. This results in more robust visual features that are less susceptible to noise in the input. The vision encoder is kept **frozen** during our training, leveraging its powerful zero-shot capabilities to extract meaningful features from the `ego_view` video stream without needing to be fine-tuned on our specific simulation data.

*   **Language Backbone (Qwen3-1.7B):** The language understanding component is a 1.7-billion parameter Qwen3 model. Its large context window (32k tokens) and sophisticated attention mechanisms (Grouped Query Attention) allow it to interpret complex, multi-step natural language instructions. This component is **fully trained** (fine-tuned) on the language commands from the dataset.

*   **Fusion Backbone (Eagle):** The vision and language features are not simply concatenated. They are intelligently fused using a cross-attention mechanism within the `EagleBackbone`. This allows the model to learn rich correlations between the visual scene and the language command (e.g., associating the words "pick up the red block" with the specific pixels corresponding to that object).

*   **Action Head (Diffusion Policy):** The final action is generated by a transformer-based diffusion policy. Instead of predicting the action directly, it starts with a tensor of random noise and iteratively refines it over 100 steps (during training) into a coherent, multi-step action sequence. This approach tends to produce smoother and more realistic trajectories than direct regression. It predicts an **8-step action horizon** for the 7 degrees of freedom of the robot arm plus the gripper state.

### Parameter and Resource Breakdown

| Component | Parameters | Memory (BF16) | Trainable | Optimization Strategy | 
| :--- | :--- | :--- | :--- | :--- | 
| SigLIP Vision | ~86M | ~172 MB | ❌ Frozen | Zero-shot transfer from pre-trained weights. | 
| Qwen3-1.7B Language | 1.7B | ~3.4 GB | ✅ Full | Fine-tuned on task data. Gradient checkpointing is critical to manage memory. | 
| EagleBackbone Fusion | ~92M | ~184 MB | ✅ Full | Trained to learn the vision-language mapping. | 
| Diffusion Action Head | ~11M | ~44 MB | ✅ Full | Trained to generate action sequences. | 
| **Total** | **~1.89B** | **~3.8 GB** | **~1.8B** | Trained with `bf16` mixed precision and distributed data parallel. | 

---

## 4. The Engineering Journey: From Failure to Success

This section details the critical engineering challenges we faced and the solutions we implemented.

### Challenge 1: The Dependency Gauntlet

Our initial attempts to run the training script were met with a cascade of environment-related failures.

*   **The `flash-attn` Problem:** The model requires `flash-attn` for efficient attention computation. However, `pip install flash-attn` consistently failed because it requires a specific combination of Python, PyTorch, and CUDA versions to find a pre-compiled binary ("wheel"). Compiling from source is notoriously difficult.
    *   **Solution:** We first identified our exact environment on the SLURM cluster: **Python 3.10.18, PyTorch 2.7.1, CUDA 12.6**. We then searched the `flash-attn` GitHub releases for a community-provided wheel that matched this stack. We found and installed `flash_attn-2.8.1+cu128torch2.4-cp310-cp310-linux_x86_64.whl`. As a fallback, we also discovered that setting the environment variable `TRANSFORMERS_NO_FLASH_ATTENTION=1` forces the Hugging Face `transformers` library to use a slower but more compatible pure-PyTorch attention implementation, which was crucial for debugging.

*   **Missing Modules:** We encountered a series of `ModuleNotFoundError` errors for `av`, `decord`, and `pytorch3d`. These are required for video processing and 3D transformations.
    *   **Solution:** We learned that dependencies must be installed *within the interactive SLURM job environment* (`srun`). Installing them on the login node is insufficient. We added `pip install av decord numpydantic` and the more complex `conda install -c fvcore -c iopath -c conda-forge fvcore iopath pytorch3d` to our setup script.

### Challenge 2: Taming the Multi-GPU Beast

With the environment fixed, we moved to multi-GPU training and faced a new set of challenges.

*   **The CUDA OOM Error:** Our initial attempts with 3 L40S GPUs and a batch size of 8 immediately resulted in CUDA Out-of-Memory errors.
    *   **Solution:** We implemented a multi-pronged memory optimization strategy:
        1.  **Mixed Precision:** We used `bf16` precision, which halves the memory footprint of model weights and gradients with minimal impact on numerical stability.
        2.  **Gradient Checkpointing:** We enabled gradient checkpointing, a technique that trades compute for memory by not storing intermediate activations during the forward pass and recomputing them during the backward pass.
        3.  **Gradient Accumulation:** We reduced the per-device batch size to a very small number (e.g., 2) and used gradient accumulation (e.g., 8 steps) to simulate a larger effective batch size (`2 * 8 * 3 GPUs = 48`) without the memory overhead.

*   **The Silent SIGTERM:** Even with memory optimizations, our `torchrun` process would sometimes be killed with `Signal 15 (SIGTERM)` without a clear CUDA error. This often indicates system-level resource exhaustion or instability in the distributed setup.
    *   **Solution:** We found that PyTorch's DDP backend is often more stable and optimized for power-of-2 GPU counts. We **reduced our training from 3 GPUs to 2 GPUs**, which resolved the instability. This highlights that using *more* GPUs is not always better if the communication overhead and topology are not optimal.

### Challenge 3: The Data-Model Mismatch

After resolving the environment and hardware issues, the training script finally launched but immediately failed during data loading.

*   **The Missing Key:** The error was `KeyError: 'right_wrist_view'`. The `GR00T` model expected multiple camera views, but our `robot_sim.PickNPlace` dataset only provided a single `ego_view`. 
    *   **Solution:** This required a deep dive into the data configuration. We created a **custom data configuration class**, `RobotSimPickNPlaceConfig`, which explicitly told the model to only expect and use the `video.ego_view` modality and to ignore all others. This custom config was registered with the framework and passed via the `--data_config` CLI argument, finally allowing the data to be loaded successfully.

---

## 5. Evaluation and Results

To validate our trained model, we developed a comprehensive evaluation script, `scripts/eval_thinkact.py`, which measures the policy's performance on a held-out test set.

### Key Metrics

| Metric | Description | Importance | 
| :--- | :--- | :--- | 
| `avg_action_mse` | The average Mean Squared Error between the predicted action sequence and the ground-truth actions from the dataset. | Measures how closely the policy imitates the expert demonstrations. Lower is better. | 
| `avg_success_rate` | The percentage of evaluation episodes where the task goal was successfully completed (requires `goal_reached` flag in dataset). | The ultimate measure of task performance. Higher is better. | 
| `avg_inference_time` | The average wall-clock time in seconds for a single forward pass of the model. | Measures the policy's suitability for real-time deployment. Lower is better. | 
| `samples_processed` | The total number of samples evaluated. | Provides context for the statistical significance of the results. | 

The script also calculates the standard deviation, min, and max for each metric, providing a complete picture of the model's performance distribution. All results are saved to a `eval_results.json` file for easy analysis.

---

## 6. Reproducibility: Your Guide to Rebuilding Gemma-Le

We are committed to open science and have made our entire workflow reproducible.

*   **Public Code Repository:** [https://github.com/Ryukijano/Gemma-Grook](https://github.com/Ryukijano/Gemma-Grook)
*   **Model Weights & Configs:** [https://huggingface.co/Ryukijano/gemma-grook](https://huggingface.co/Ryukijano/gemma-grook)

To reproduce our results, follow these steps:

1.  **Clone the Repository:** `git clone https://github.com/Ryukijano/Gemma-Grook.git`
2.  **Set up the Environment:** Use the provided `pyproject.toml` or an environment YAML file to create a Conda environment with the exact dependencies we used.
3.  **Download Data:** Use the `lerobot` CLI to download the `robot_sim.PickNPlace` dataset.
4.  **Run Training:** Execute the provided `scripts/run_training_multi_gpu.sh` script, which contains the `accelerate launch` command with all the correct arguments and environment variables.
5.  **Run Evaluation:** Once training is complete, use the `scripts/run_evaluation.sh` script, pointing it to your checkpoint directory, to generate the final performance metrics.

---

## 7. Conclusion

Gemma-Le demonstrates that training a large, state-of-the-art VLA model is achievable, but success depends heavily on overcoming a series of practical engineering hurdles that are rarely discussed in academic literature. By documenting our solutions to dependency conflicts, multi-GPU instability, and data-model mismatches, we provide a valuable, reproducible resource for the robotics community. Our work lowers the barrier to entry for training large policies and serves as a practical guide for future research in AI-driven robotic manipulation.
