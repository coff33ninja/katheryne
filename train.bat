@echo off
echo Starting Genshin Assistant Training...
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Check if required packages are installed
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Error: PyTorch is not installed
    echo Installing required packages...
    pip install -r python/requirements.txt
)

:: Run hardware detection
echo Running hardware detection...
python python/check_hardware.py

:: Set environment variables for training (can be modified as needed)
set EPOCHS=10
set BATCH_SIZE=64
set LEARNING_RATE=0.002

echo Training Configuration:
echo Epochs: %EPOCHS%
echo Batch Size: %BATCH_SIZE%
echo Learning Rate: %LEARNING_RATE%
echo.

:: Run the training script
echo Starting training...
python python/train_assistant.py

:: Check if training completed successfully
if errorlevel 1 (
    echo.
    echo Training failed with an error
    pause
    exit /b 1
) else (
    echo.
    echo Training completed successfully!
    pause
)
