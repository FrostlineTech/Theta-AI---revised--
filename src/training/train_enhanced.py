"""
Enhanced training script for Theta AI with early stopping, improved learning rate schedule, 
and validation-based model saving.
"""

import os
import json
import torch
import argparse
import numpy as np
import math
import time
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, GPT2TokenizerFast, GPT2LMHeadModel
from tqdm.auto import tqdm
import sys
from colorama import Fore, Back, Style, init
import logging
import shutil

# Initialize colorama for Windows support
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path to import model
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)
from src.model.theta_model import ThetaModel

class ThetaDataset(Dataset):
    """Custom dataset for Theta AI training."""
    
    def __init__(self, data_path, model_name, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the processed data JSON file
            model_name: Name of the model to use
            tokenizer: Tokenizer for encoding the data
            max_length: Maximum sequence length
        """
        # Specify model architecture and tokenizer
        try:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Error loading model: {str(e)}")
            logger.info(f"Attempting to download {model_name} from Hugging Face...")
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        
        # Set special tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.sep_token = '<|sep|>'
        if self.tokenizer.sep_token not in self.tokenizer.get_vocab():
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        logger.info(f"Loaded {model_name} with {sum(p.numel() for p in self.model.parameters())/1000000:.2f}M parameters")
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        # Prepare formatted data for GPT-2 style training
        self.formatted_data = []
        for item in self.data:
            question = item['question']
            answer = item['answer']
            
            # Format as instruction-following text
            text = f"Question: {question}\nAnswer: {answer}\n\n"
            self.formatted_data.append(text)
            
        logger.info(f"Loaded {len(self.formatted_data)} training examples")
    
    def __len__(self):
        return len(self.formatted_data)
    
    def __getitem__(self, idx):
        text = self.formatted_data[idx]
        
        # Tokenize text
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create language modeling labels (same as input_ids)
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # For GPT-2 training, labels are the same as input_ids
        labels = input_ids.clone()
        
        # Set labels for padding tokens to -100 so they're ignored in the loss
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class EarlyStopping:
    """Early stopping implementation to prevent overfitting."""
    
    def __init__(self, patience=3, min_delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as improvement
            path: Path to save the best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss  # Higher score is better (negative loss)
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        logger.info(f"Validation loss decreased to {val_loss:.6f}. Saving model...")
        torch.save(model.state_dict(), self.path)

def train(args):
    """
    Train the Theta AI model with enhanced features.
    
    Args:
        args: Training arguments
    """
    # Set default gradient accumulation steps if not provided
    if not hasattr(args, 'gradient_accumulation_steps'):
        args.gradient_accumulation_steps = 4
        
    # Log all training parameters
    logger.info(f"Training Theta AI model with the following settings:")
    logger.info(f"- Model type: {args.model_type}")
    logger.info(f"- Model name: {args.model_name}")
    logger.info(f"- Data path: {args.data_path}")
    logger.info(f"- Output dir: {args.output_dir}")
    logger.info(f"- Batch size: {args.batch_size}")
    logger.info(f"- Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"- Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"- Learning rate: {args.learning_rate}")
    logger.info(f"- Epochs: {args.epochs}")
    logger.info(f"- Patience: {args.patience}")
    logger.info(f"- Warmup proportion: {args.warmup_proportion}")
    logger.info(f"- Weight decay: {args.weight_decay}")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    # Initialize model
    theta = ThetaModel(model_type=args.model_type, model_name=args.model_name, device=device)
    
    # Apply GPU optimizations
    theta.optimize_for_gpu()
    
    # Create dataset and dataloader
    dataset = ThetaDataset(args.data_path, args.model_name, theta.tokenizer)
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Set up optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in theta.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in theta.model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Calculate total training steps and warmup steps
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_proportion)
    
    # Learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        path=os.path.join(args.output_dir, "best_model.pt")
    )
    
    # Keep track of best validation loss
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Helper functions for colorful output
    def print_header(text):
        print(f"\n{Fore.CYAN}{Back.BLACK}{Style.BRIGHT}== {text} =={Style.RESET_ALL}")
        
    def print_metric(name, value, color=Fore.GREEN, is_good=True):
        trend = "↓" if is_good and len(train_losses) > 1 and value < train_losses[-2] else "↑"
        trend = Fore.GREEN + trend if is_good else Fore.RED + trend
        print(f"{color}{name}: {value:.4f} {trend}{Style.RESET_ALL}")

    # Save a copy of the script for reproducibility
    script_path = os.path.abspath(__file__)
    script_backup = os.path.join(args.output_dir, "training_script_backup.py")
    shutil.copy2(script_path, script_backup)
    logger.info(f"Saved script backup to {script_backup}")
    
    # Save training configuration
    config_backup = os.path.join(args.output_dir, "training_config.json")
    with open(config_backup, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved training configuration to {config_backup}")
    
    # Start time for the training run
    training_start_time = time.time()
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print_header(f"Epoch {epoch+1}/{args.epochs}")
        theta.model.train()
        
        # Use mixed precision training for RTX 3060 optimization
        scaler = torch.cuda.amp.GradScaler()
        
        total_train_loss = 0
        batch_count = len(train_dataloader)
        
        # Training phase
        for i, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = theta.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # Normalize loss for gradient accumulation
                loss = outputs.loss / args.gradient_accumulation_steps
                
            # Backward pass with scaling for mixed precision
            scaler.scale(loss).backward()
            
            # Only update weights after accumulating enough gradients
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(theta.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Zero gradients after optimizer step
                
                # Update learning rate scheduler
                scheduler.step()
            
            # Update progress
            current_loss = loss.item()
            total_train_loss += current_loss
            
            # Print progress every 10 batches or at the end
            if (i + 1) % 10 == 0 or (i + 1) == batch_count:
                print(f"\rProgress: {i+1}/{batch_count} batches - Loss: {current_loss:.4f}", end="")
        
        # Calculate average training loss for this epoch
        avg_train_loss = total_train_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"\nEpoch {epoch+1} - Avg training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        theta.model.eval()
        total_val_loss = 0
        batch_count_val = len(val_dataloader)
        
        print_header("Validation")
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = theta.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    val_loss = outputs.loss
                    
                total_val_loss += val_loss.item()
                
                # Print progress every 10 batches or at the end
                if (i + 1) % 10 == 0 or (i + 1) == batch_count_val:
                    print(f"\rValidating: {i+1}/{batch_count_val} batches", end="")
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / batch_count_val
        val_losses.append(avg_val_loss)
        print(f"\nEpoch {epoch+1} - Validation loss: {avg_val_loss:.4f}")
        
        # Calculate perplexity
        train_perplexity = math.exp(avg_train_loss)
        val_perplexity = math.exp(avg_val_loss)
        
        # Print metrics summary
        print_header("Metrics Summary")
        print_metric("Training Loss", avg_train_loss, Fore.GREEN, is_good=True)
        print_metric("Validation Loss", avg_val_loss, Fore.YELLOW, is_good=True)
        print_metric("Training Perplexity", train_perplexity, Fore.CYAN, is_good=True)
        print_metric("Validation Perplexity", val_perplexity, Fore.MAGENTA, is_good=True)
        
        # Print improvement percentage if applicable
        if len(val_losses) > 1:
            prev_loss = val_losses[-2]
            improvement = (prev_loss - avg_val_loss) / prev_loss * 100
            direction = "improved" if improvement > 0 else "worsened"
            color = Fore.GREEN if improvement > 0 else Fore.RED
            print(f"{color}Validation loss {direction} by {abs(improvement):.2f}%{Style.RESET_ALL}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        print(f"{Fore.BLUE}Epoch completed in {epoch_time:.2f} seconds{Style.RESET_ALL}")
        
        # Save regular checkpoint
        checkpoint_dir = os.path.join(
            args.output_dir, 
            f"theta_checkpoint_epoch_{epoch+1}"
        )
        
        theta.save(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Check for early stopping
        early_stopping(avg_val_loss, theta.model)
        if early_stopping.early_stop:
            print(f"{Fore.YELLOW}Early stopping triggered! No improvement for {args.patience} epochs.{Style.RESET_ALL}")
            break
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        theta.model.load_state_dict(torch.load(best_model_path))
        logger.info("Loaded best model based on validation loss")
    
    # Save final model
    final_model_dir = os.path.join(args.output_dir, "theta_final")
    theta.save(final_model_dir)
    logger.info(f"Training complete! Final model saved to {final_model_dir}")
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Print final performance summary
    print_header("Final Performance Summary")
    print(f"{Fore.CYAN}Initial training loss: {train_losses[0]:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Final training loss: {train_losses[-1]:.4f}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Best validation loss: {min(val_losses):.4f}{Style.RESET_ALL}")
    
    # Calculate overall improvement
    if len(train_losses) > 1:
        total_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"{Fore.GREEN}Overall loss improvement: {total_improvement:.2f}%{Style.RESET_ALL}")
    
    # Save learning curves
    loss_data = {
        'train_loss': train_losses,
        'val_loss': val_losses,
    }
    with open(os.path.join(args.output_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    return final_model_dir

def main(passed_args=None):
    # Only parse arguments if not passed in
    if passed_args is None:
        parser = argparse.ArgumentParser(description="Train the Theta AI model with enhanced features")
        
        parser.add_argument("--model_type", type=str, default="gpt2", help="Model type (gpt2, bert-qa)")
        parser.add_argument("--model_name", type=str, default="gpt2", help="Specific model name/version")
        parser.add_argument("--data_path", type=str, required=True, help="Path to the processed data")
        parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
        parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                          help="Number of steps to accumulate gradients before performing a backward/update pass")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
        parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
        parser.add_argument("--warmup_proportion", type=float, default=0.1, 
                          help="Proportion of training steps for learning rate warmup")
        parser.add_argument("--weight_decay", type=float, default=0.01, 
                          help="Weight decay for regularization")
        parser.add_argument("--no-color", action="store_true", help="Disable colored output")
        parser.add_argument("--log_file", type=str, help="Path to log file")
        
        args = parser.parse_args()
    else:
        args = passed_args
    
    # Set up file logging if specified
    if hasattr(args, 'log_file') and args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Disable colors if requested
    if hasattr(args, 'no_color') and args.no_color:
        init(autoreset=True, strip=True)
    else:
        init(autoreset=True)
        
    # Print colorful banner
    print(f"{Fore.CYAN}{Style.BRIGHT}╔════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}║     THETA AI ENHANCED TRAINING SYSTEM v2.0     ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}╚════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    train(args)

if __name__ == "__main__":
    main()
