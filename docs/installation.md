# Installation Guide for Theta AI

This guide provides detailed instructions for installing and setting up the Theta AI environment.

## System Requirements

### Hardware Requirements

- **CPU**: AMD Ryzen 5-5500 or equivalent (6 cores / 12 threads recommended)
- **GPU**: NVIDIA RTX 3060 with 12GB VRAM
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB minimum for code, datasets, and model checkpoints

### Software Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04 LTS or newer
- **CUDA Toolkit**: Version 11.7 or newer
- **Python**: Version 3.8 or newer
- **Git**: Latest version

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>/theta-ai.git
cd theta-ai
```

### 2. Set Up Python Environment

#### Using venv (recommended)

```bash
python -m venv theta-env
```

Activate the virtual environment:

**Windows**:

```bash
theta-env\Scripts\activate
```

**Linux/macOS**:

```bash
source theta-env/bin/activate
```

#### Using Conda

```bash
conda create -n theta-env python=3.8
conda activate theta-env
```

### 3. Install Dependencies

Install core dependencies:

```bash
pip install -r requirements.txt
```

For web interface functionality:

```bash
pip install -r web_requirements.txt
```

### 4. Install CUDA Toolkit

Download and install the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

Verify installation:

```bash
nvcc --version
```

### 5. Download or Create Model Checkpoints

#### Option A: Download Pre-trained Model (if available)

Download the pre-trained model checkpoints from your organization's model repository and place them in the `models/` directory.

#### Option B: Train a New Model

Run the training script to create a new model:

```bash
train_overnight.bat
```

Or for Linux:

```bash
bash train_overnight.sh
```

## Configuration

### Environment Configuration

Create a `.env` file in the root directory with the following settings:

```env
MODEL_PATH=models/theta_checkpoint_epoch_26
FALLBACK_MODEL_PATH=models/theta_latest
DATASET_DIR=Datasets
LOG_LEVEL=INFO
```

### Web Interface Configuration

If using the web interface, review and adjust the settings in `webserver_config.json`.

## Verify Installation

To verify that everything is working correctly:

1. Run the test script:

   ```bash
   python src/test_installation.py
   ```

2. Start the command-line interface:

   ```bash
   interface.bat
   ```

3. Try a sample query like: "What is the difference between AMD and NVIDIA GPUs?"

## Upgrading

To upgrade Theta AI:

1. Pull the latest changes:

   ```bash
   git pull origin main
   ```

2. Update dependencies:

   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. Run the upgrade script to handle any configuration changes:

   ```bash
   python upgrade_theta.py
   ```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

- Verify CUDA installation with `nvidia-smi`
- Ensure PyTorch is installed with CUDA support
- Check compatibility between PyTorch and CUDA versions

### Memory Errors

For "CUDA out of memory" errors:

- Reduce batch size in training scripts
- Close other GPU-intensive applications
- Enable gradient accumulation

### Import Errors

For Python import errors:

- Check that all dependencies are installed
- Verify virtual environment is activated
- Check for version conflicts with `pip list`

## Next Steps

After completing installation:

- See [Getting Started](./getting_started.md) for initial usage
- Review [Training Guide](./training_guide.md) for model training
- Explore [Dataset Guide](./dataset_guide.md) for data preparation
