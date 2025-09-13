@echo off
echo Theta AI - Overnight Training Session
echo ====================================
echo.
echo This script will run an extended training session optimized for RTX 3060
echo Started at: %date% %time%
echo.

REM Set environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=.

REM Create training log directory if it doesn't exist
if not exist "logs" mkdir logs

REM Create timestamped log file
set logfile=logs\training_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set logfile=%logfile: =0%

echo Starting extended training session...
echo Training log: %logfile%
echo.

REM Run the training with extended epochs and optimized parameters for 12GB VRAM
REM Display colorful output directly in the console
python src/train.py ^
  --data_path "Datasets/processed_data.json" ^
  --output_dir "models" ^
  --model_name "gpt2-medium" ^
  --batch_size 2 ^
  --gradient_accumulation_steps 8 ^
  --learning_rate 2e-5 ^
  --epochs 100

echo.
echo Training completed at: %date% %time%
echo.
echo To use the trained model, run: interface.bat
echo.

pause
