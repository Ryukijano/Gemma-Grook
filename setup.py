from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="isaac-gr00t",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Gemma-based robot learning with LeRobot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/isaac-gr00t",
    packages=find_packages(include=["lerobot", "lerobot.*"]),
    package_data={"lerobot": ["configs/*.yaml", "configs/**/*.yaml"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pytest>=7.3.1",
            "pytest-cov>=4.0.0",
            "sphinx>=6.1.3",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gr00t-train=scripts.gr00t_finetune:main",
        ],
    },
)
