# Development Guide

This guide provides instructions for setting up the development environment and contributing to the project.

## Prerequisites

- Python 3.10+
- CUDA 12.4+
- NVIDIA GPU (L40S or compatible)
- Git

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/isaac-gr00t.git
   cd isaac-gr00t
   ```

2. **Create and activate a conda environment**
   ```bash
   conda create -n gr00t python=3.10
   conda activate gr00t
   ```

3. **Install PyTorch with CUDA support**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Install project dependencies**
   ```bash
   pip install -e .[dev]
   ```

## Project Structure

```
isaac-gr00t/
├── lerobot/               # Core library code
│   ├── common/
│   │   └── policies/     # Policy implementations
│   └── configs/          # Configuration files
├── scripts/              # Training and evaluation scripts
├── tests/                # Test files
├── demo_data/            # Example datasets
└── output/               # Training outputs and checkpoints
```

## Development Workflow

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style (black, isort)
   - Add tests for new features
   - Update documentation as needed

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Submit a pull request**
   - Push your branch to GitHub
   - Create a pull request against the main branch
   - Request reviews from team members

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for all function signatures
- Keep lines under 100 characters
- Use docstrings for all public functions and classes

## Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_module.py

# Run tests with coverage
pytest --cov=lerobot tests/
```

## Building Documentation

```bash
cd docs
make html
```

## Release Process

1. Update version in `__init__.py`
2. Update `CHANGELOG.md`
3. Create a release tag
   ```bash
   git tag -a v0.1.0 -m "Initial release"
   git push origin v0.1.0
   ```
4. Create a GitHub release with release notes
