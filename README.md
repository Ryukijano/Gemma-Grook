# Gemma‑Le (Gemma‑GR00T): SigLIP + Gemma 3 + ScaleDP for Robotic Manipulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)


Gemma‑Le is an open-source Vision‑Language‑Action policy built on LeRobot and GR00T. It replaces prior NV Eagle components with standard Hugging Face backbones and a diffusion action head:

- Vision: SigLIP (`google/siglip-so400m-patch14-384`)
- Language: Gemma 3 (`google/gemma-3-4b-it`) with LoRA on q/k/v/o proj (rank=16)
- Action head: ScaleDP Transformer (layers=12, d_model=320, heads=8, ff=1280)

The policy fuses SigLIP and Gemma features into a `conditioning_dim=768` vector and trains a diffusion head to predict action noise over `num_diffusion_steps=50` with temporal context `chunk_size=8`.
=======
Gemma-GR00T is an open-source project that integrates Google's Gemma language model with NVIDIA's GR00T robotics framework to create advanced multimodal vision-language-action policies for robotic manipulation tasks. This project enables robots to understand natural language instructions, perceive their environment through vision, and perform precise manipulation tasks.
![Gemma-Gr00t](https://github.com/Ryukijano/Gemma-Grook/blob/main/84c6e56a-e60e-4d9b-a745-c5f6976ce746.png)

## 🚀 Features

- **Multimodal Integration**: Combines vision, language, and action in a unified framework
- **Gemma Language Model**: Leverages Google's state-of-the-art language understanding
- **GR00T Framework**: Built on NVIDIA's robust robotics infrastructure
- **Flexible Training**: Supports both simulation and real-world robot training
- **Scalable Architecture**: Designed for single-GPU development and multi-GPU scaling

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA 12.4+
- PyTorch 2.7.1+
- NVIDIA GPU with at least 48GB VRAM (for full training)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Ryukijano/Gemma-Grook.git
   cd Gemma-Grook
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n gemma-groot python=3.10
   conda activate gemma-groot
   ```

3. Install dependencies:
   ```bash
   pip install -e "."
   ```

4. (Optional) Install flash-attention for better performance:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

## 🛠️ Usage

### Training (LeRobot entrypoint)

Example single-node training:
```bash
python lerobot/lerobot/scripts/train.py \
  --policy.type gemma_le \
  --dataset.repo_id local/robot_sim.PickNPlace \
  --dataset.root /scratch/cbjp404/Isaac-GR00T/demo_data/robot_sim.PickNPlace \
  --dataset.episodes "[0,1,2,3,4]" \
  --batch_size 3 \
  --steps 200000 \
  --log_freq 100 \
  --save_freq 5000 \
  --policy.vision_model_id google/siglip-so400m-patch14-384 \
  --policy.text_model_id google/gemma-3-4b-it \
  --policy.use_amp true \
  --progress_bar true
```

Slurm (3× L40) example: see `submit_job.sh`.

### Export and Upload

Export weights and upload to Hugging Face Hub:
```bash
python scripts/export_weights.py --checkpoint <checkpoint_dir>
python scripts/upload_to_hf.py --model_path exported_weights/<run> --repo_id Ryukijano/gemma-groot --private
```

### SLURM Job Submission

Use the provided SLURM script for cluster execution:
```bash
sbatch submit_job.sh
```

## 📂 Project Structure (key)

```
Gemma-GR00T/
├── gr00t/                    # Core package
│   ├── eval/                 # Evaluation utilities
│   ├── experiment/           # Experiment configuration
│   ├── model/                # Model definitions
│   └── utils/                # Utility functions
├── lerobot/                  # Vendored LeRobot fork (datasets, train loop, policies)
├── scripts/                  # Training/eval/export + HF upload helpers
├── tests/                    # Unit tests
├── .gitignore               # Git ignore file
├── pyproject.toml           # Project metadata
└── requirements.txt         # Python dependencies
```

## 📚 Docs & Model Card

For more details, see the Hub model card `Ryukijano/gemma-groot` and:

- [GR00T Framework Documentation](https://developer.nvidia.com/robotics/groot)
- [Gemma Model Card](https://ai.google.dev/gemma)
- [LeRobot](https://huggingface.co/lerobot)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NVIDIA GR00T](https://developer.nvidia.com/robotics/groot) - Robotics framework
- [Google Gemma](https://ai.google.dev/gemma) - Language model
- [Hugging Face](https://huggingface.co/) - Model hub and tools
- [LeRobot](https://huggingface.co/lerobot) - Robotics dataset and tools
