#!/usr/bin/env python3
"""
Setup script for document-classifier package.
Run: pip install -e . 
This will install your project as a package in development mode.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="document-classifier",
    version="1.0.0",
    description="Document image classification with PyTorch and Hydra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/document-classifier",
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "timm>=0.5.4",
        "transformers>=4.12.0",
        "numpy>=1.22.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.2",
        "Pillow>=8.3.0",
        "opencv-python>=4.5.5",
        "albumentations>=1.1.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
        "fire>=0.7.0",
        "wandb>=0.16.0",
        "hydra-core>=1.3.0",
        "hydra-colorlog>=1.2.0",
        "omegaconf>=2.3.0",
    ],
    
    # Development dependencies
    extras_require={
        "dev": [
            "jupyter",
            "ipywidgets",
            "pytest",
            "black",
            "flake8",
        ]
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "doc-train=scripts.train:main",
            "doc-predict=scripts.predict:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)