THU Data-Mining Coursework

# Environment Setup for Bank Marketing Classification

This document describes how to set up the environment for running the bank marketing classification scripts.

## Installation Methods

### Option 1: Using pip (recommended for most users)

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda (recommended for data science)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate the environment
conda activate data-mining-env
```

## Platform-Specific PyTorch Installation

The scripts support GPU acceleration if available. For optimal performance:

### macOS (Apple Silicon - M1/M2/M3)
```bash
# PyTorch with MPS (Metal Performance Shaders) support
pip install torch torchvision
```

### Linux/Windows with NVIDIA GPU
```bash
# PyTorch with CUDA support (check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

0000
