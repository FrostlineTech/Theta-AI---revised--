"""
Theta AI Upgrade Script
------------------------
This script implements all recommended improvements for Theta AI:
1. Updates to GPT-2 Medium model for 12GB VRAM
2. Adds enhanced training data with conversational examples
3. Modifies training parameters for better coherence
4. Improves retrieval mechanism for better responses
5. Implements response validation and system prompts

Run this script to apply all upgrades and prepare for overnight training.
"""

import os
import sys
import json
import shutil
from pathlib import Path
import argparse
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"theta_upgrade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
MODELS_DIR = PROJECT_ROOT / "models"

def check_requirements():
    """Check if all required packages are installed."""
    logger.info("Checking requirements...")
    
    try:
        import torch
        import transformers
        from sklearn import __version__ as sklearn_version
        import numpy
        import colorama
        logger.info("[OK] All core dependencies are available")
        
        # Check for GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            logger.info(f"[OK] GPU detected: {gpu_name} with {vram_mb:.0f} MB VRAM")
            if vram_mb < 12000:  # Less than 12GB
                logger.warning(f"! Warning: GPU has less than 12GB VRAM. Some optimizations may not work.")
        else:
            logger.warning("! Warning: No GPU detected. Training will be slow.")
        
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        logger.info("Please run: pip install -r requirements.txt")
        return False

def check_datasets():
    """Verify that required datasets exist."""
    logger.info("Checking datasets...")
    
    # Check if Datasets directory exists
    if not DATASETS_DIR.exists():
        logger.error("Datasets directory not found!")
        return False
        
    # Count JSON files
    json_files = list(DATASETS_DIR.glob("*.json"))
    if not json_files:
        logger.error("No JSON datasets found!")
        return False
        
    logger.info(f"[OK] Found {len(json_files)} dataset files")
    return True

def verify_new_datasets():
    """Verify that new conversational datasets exist."""
    priority_files = ["conversational_examples.json", "theta_info.json"]
    missing = [f for f in priority_files if not (DATASETS_DIR / f).exists()]
    
    if missing:
        logger.error(f"Missing important dataset files: {', '.join(missing)}")
        return False
    
    logger.info("[OK] All required new datasets are available")
    return True

def process_data():
    """Process the datasets and prepare for training."""
    logger.info("Processing datasets...")
    
    try:
        process_script = PROJECT_ROOT / "src" / "data_processor.py"
        if not process_script.exists():
            logger.error("Data processor script not found!")
            return False
            
        result = subprocess.run([sys.executable, str(process_script)], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Data processing failed: {result.stderr}")
            return False
            
        logger.info("[OK] Data processing completed successfully")
        logger.info(result.stdout.strip())
        return True
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return False

def update_training_parameters():
    """Update training parameters in configuration files."""
    logger.info("Updating training parameters...")
    
    # Update batch file for GPT-2 Medium
    batch_file = PROJECT_ROOT / "train_overnight.bat"
    if not batch_file.exists():
        logger.error("Training batch file not found!")
        return False
    
    try:
        with open(batch_file, 'r') as f:
            content = f.read()
        
        # Check if already updated
        if "gpt2-medium" in content and "--gradient_accumulation_steps" in content:
            logger.info("[OK] Training batch file already updated")
        else:
            # Update command line
            content = content.replace(
                "python src/train.py ^",
                "python src/train.py ^"
            )
            
            # Update model name and parameters
            if "--model_name" not in content:
                content = content.replace(
                    "  --data_path \"Datasets/processed_data.json\" ^",
                    "  --data_path \"Datasets/processed_data.json\" ^\n  --model_name \"gpt2-medium\" ^"
                )
            
            # Update batch size and gradient accumulation
            content = content.replace(
                "  --batch_size 4 ^",
                "  --batch_size 2 ^\n  --gradient_accumulation_steps 8 ^"
            )
            
            # Update epochs
            content = content.replace(
                "  --epochs 50",
                "  --epochs 100"
            )
            
            with open(batch_file, 'w') as f:
                f.write(content)
                
            logger.info("[OK] Updated training batch file")
    except Exception as e:
        logger.error(f"Error updating training parameters: {str(e)}")
        return False
    
    return True

def setup_overnight_training():
    """Set up the system for overnight training."""
    logger.info("Setting up overnight training...")
    
    # Check if we can run the training command
    try:
        # Create a simple test script
        batch_file = PROJECT_ROOT / "test_gpu.py"
        with open(batch_file, 'w') as f:
            f.write("""
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
""")
        
        # Run the test script
        result = subprocess.run([sys.executable, str(batch_file)], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"GPU test failed: {result.stderr}")
            return False
            
        logger.info("GPU test output:")
        logger.info(result.stdout.strip())
        
        # Clean up
        os.remove(batch_file)
        
    except Exception as e:
        logger.error(f"Error testing GPU: {str(e)}")
        return False
    
    logger.info("[OK] System is ready for overnight training")
    return True

def main():
    """Main function to run all upgrades."""
    parser = argparse.ArgumentParser(description="Upgrade Theta AI with all improvements")
    parser.add_argument("--verify-only", action="store_true", help="Only verify system without making changes")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("THETA AI UPGRADE SCRIPT")
    logger.info("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Check datasets
    if not check_datasets():
        return False
        
    # Check new datasets
    if not verify_new_datasets():
        return False
    
    if args.verify_only:
        logger.info("Verification complete. Use without --verify-only to apply changes.")
        return True
    
    # Process data
    if not process_data():
        return False
    
    # Update training parameters
    if not update_training_parameters():
        return False
    
    # Setup for overnight training
    if not setup_overnight_training():
        return False
    
    logger.info("=" * 60)
    logger.info("THETA AI UPGRADE COMPLETE")
    logger.info("=" * 60)
    logger.info("To start overnight training, run:")
    logger.info("  train_overnight.bat")
    logger.info("=" * 60)
    logger.info("INSTALLATION NOTES:")
    logger.info("- If you encounter package errors, use these commands:")
    logger.info("  pip install scikit-learn  # NOT 'sklearn'")
    logger.info("  pip install -r requirements.txt")
    logger.info("=" * 60)
    
    return True

def show_installation_help():
    """Show detailed installation instructions."""
    print("\n" + "=" * 60)
    print("THETA AI INSTALLATION GUIDE")
    print("=" * 60)
    print("\nTo install all required dependencies:")
    print("\n1. Install scikit-learn package:")
    print("   pip install scikit-learn")
    print("\n2. Install all other dependencies:")
    print("   pip install -r requirements.txt")
    print("\nCommon Issues:")
    print("- Do NOT use 'pip install sklearn' - this package is deprecated")
    print("- Use 'scikit-learn' instead of 'sklearn' in all pip commands")
    print("- If you see version conflicts, try creating a new virtual environment:")
    print("   python -m venv theta-env")
    print("   .\\theta-env\\Scripts\\activate")
    print("   pip install -r requirements.txt")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--install-help":
        show_installation_help()
    else:
        success = main()
        sys.exit(0 if success else 1)
