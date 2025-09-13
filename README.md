# Theta AI

Theta is an internal AI assistant for Frostline Solutions employees, designed to help with cybersecurity, software development, and general company information.

## Features

- Trained on Frostline-specific information with enhanced conversational capabilities
- Optimized for RTX 3060 GPU (12GB VRAM)
- Advanced retrieval mechanism with TF-IDF similarity for more relevant responses
- Mobile-friendly web interface
- Robust response validation and hallucination prevention
- Can answer questions about:
  - Frostline Solutions services and company information
  - Cybersecurity concepts and best practices
  - Software development guidelines
  - Cloud computing and IT infrastructure
  - Technical support

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Getting Started Guide](docs/getting_started.md)
- [Architecture Overview](docs/architecture.md)
- [Training Guide](docs/training_guide.md)
- [Dataset Guide](docs/dataset_guide.md)
- [Full Documentation Index](docs/README.md)

For contributors:
- [Contributing Guidelines](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Getting Started

### Requirements

- NVIDIA RTX 3060 GPU (12GB)
- AMD Ryzen 5-5500
- CUDA toolkit
- Python 3.8+

### Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For web interface dependencies:

```bash
pip install -r web_requirements.txt
```

### Usage

#### Training

For overnight training with optimal parameters for RTX 3060:

```bash
train_overnight.bat
```

For enhanced training with additional optimizations:

```bash
train_overnight_enhanced.bat
```

See the [Training Guide](docs/training_guide.md) for detailed information on training options and parameters.

#### Using Theta AI

To use the command-line interface:

```bash
interface.bat
```

To start the web interface:

```bash
web_interface.bat
```

Both interfaces automatically use the best model checkpoint (epoch 26) with the lowest validation loss for optimal responses. If this checkpoint isn't available, they will fall back to the latest available checkpoint or the final model.

## Project Structure

- `datasets/`: Contains training data in JSON format
- `docs/`: Comprehensive documentation
- `logs/`: Training and interface logs
- `models/`: Saved model checkpoints and configurations
- `src/`: Source code
  - `data_processing/`: Scripts for data preparation
  - `model/`: Model architecture definition
  - `training/`: Training pipelines
  - `interface/`: User interaction interfaces
- `static/`: Static assets for web interface
- `templates/`: HTML templates for web interface

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

## Version History

See the [Changelog](CHANGELOG.md) for version history and release notes.
