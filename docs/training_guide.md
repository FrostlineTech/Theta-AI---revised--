# Training Guide for Theta AI

This guide provides detailed instructions on how to train, fine-tune, and evaluate the Theta AI model.

## Prerequisites

Before starting the training process, ensure you have:

- NVIDIA RTX 3060 GPU with 12GB VRAM
- CUDA toolkit properly installed and configured
- All dependencies installed: `pip install -r requirements.txt`
- Prepared datasets in the correct format (see [Dataset Guide](./dataset_guide.md))

## Quick Start: Using the Training Script

The easiest way to start training is with the provided batch script:

```bash
train_overnight.bat
```

This script will:

- Configure optimal training parameters for RTX 3060
- Train for 100 epochs using gradient accumulation
- Save checkpoints after each epoch
- Log training metrics to the `logs/` directory

## Manual Training

For more control over the training process:

```bash
python src/train.py --epochs 100 --batch-size 4 --grad-accum 4 --learning-rate 5e-5 --checkpoint-dir models/theta_custom
```

## Enhanced Training with Advanced Features

For training with additional optimizations:

```bash
train_overnight_enhanced.bat
```

This uses improved training techniques including:

- Mixed precision training
- Enhanced data augmentation
- Dynamic learning rate scheduling

## Training Parameters

Key parameters to consider when training:

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `--epochs` | Number of training epochs | 100 |
| `--batch-size` | Batch size per gradient update | 4 |
| `--grad-accum` | Gradient accumulation steps | 4 |
| `--learning-rate` | Initial learning rate | 5e-5 |
| `--warmup-steps` | Learning rate warmup steps | 500 |
| `--weight-decay` | Weight decay for AdamW | 0.01 |
| `--max-seq-length` | Maximum sequence length | 512 |

## Checkpoints and Model Saving

During training:

- Checkpoints are saved to the specified directory (default: `models/`)
- Each checkpoint includes model weights, optimizer state, and training metrics
- The best model (lowest validation loss) is saved separately

## Training Monitoring

Monitor training progress:

```bash
tensorboard --logdir=logs
```

This will start a TensorBoard server where you can view:

- Training and validation loss
- Learning rate schedule
- Model gradients
- Example predictions

## Fine-tuning Existing Models

To fine-tune an existing checkpoint:

```bash
python src/train.py --epochs 20 --batch-size 4 --load-checkpoint models/theta_checkpoint_epoch_26
```

## Evaluation

Evaluate model performance:

```bash
python src/evaluate.py --model-path models/theta_checkpoint_epoch_26
```

This will generate:

- Accuracy metrics
- ROUGE and BLEU scores
- Detailed error analysis
- Example responses

## Optimizing for RTX 3060

The RTX 3060 has 12GB VRAM, which requires specific optimization:

1. Use gradient accumulation (4-8 steps) to simulate larger batch sizes
2. Enable mixed precision training to reduce memory usage
3. Consider sequence length carefully to balance context and memory usage
4. Monitor GPU memory usage during training using `nvidia-smi`

## Troubleshooting

Common training issues:

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, enable gradient accumulation |
| Slow convergence | Adjust learning rate, increase training epochs |
| Overfitting | Increase dropout, add regularization, use early stopping |
| Validation loss spikes | Reduce learning rate, check for data issues |

## Advanced: Distributed Training

For multi-GPU setups or future expansion:

```bash
python -m torch.distributed.launch --nproc_per_node=2 src/train.py --distributed
```
