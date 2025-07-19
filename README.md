# Gemma-GR00T: Multimodal Robotic Manipulation with Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Gemma-GR00T is an open-source project that integrates Google's Gemma language model with NVIDIA's GR00T robotics framework to create advanced multimodal vision-language-action policies for robotic manipulation tasks. This project enables robots to understand natural language instructions, perceive their environment through vision, and perform precise manipulation tasks.

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

### Training

To start training with default parameters:
```bash
python scripts/gr00t_finetune.py
```

For multi-GPU training:
```bash
torchrun --nproc_per_node=2 scripts/gr00t_finetune.py
```

### Evaluation

Run the evaluation script:
```bash
python scripts/eval_policy.py --checkpoint <path_to_checkpoint>
```

### SLURM Job Submission

Use the provided SLURM script for cluster execution:
```bash
sbatch submit_job.sh
```

## 📂 Project Structure

```
Gemma-GR00T/
├── gr00t/                    # Core package
│   ├── eval/                 # Evaluation utilities
│   ├── experiment/           # Experiment configuration
│   ├── model/                # Model definitions
│   └── utils/                # Utility functions
├── scripts/                  # Training and evaluation scripts
├── tests/                    # Unit tests
├── .gitignore               # Git ignore file
├── pyproject.toml           # Project metadata
└── requirements.txt         # Python dependencies
```

## 📚 Documentation

For detailed documentation, please refer to:

- [GR00T Framework Documentation](https://developer.nvidia.com/robotics/groot)
- [Gemma Model Card](https://ai.google.dev/gemma)
- [LeRobot Integration Guide](https://huggingface.co/lerobot)

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