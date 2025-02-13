@echo off
setlocal enabledelayedexpansion

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

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
echo Installing dependencies...
pip install -r requirements.txt

echo Setup completed successfully!

:: Check if required packages are installed
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Error: PyTorch is not installed
    echo Installing required packages...
    pip install torch tqdm
)

echo Running hardware detection...
python -c "import torch; def check_hardware(): print('\nHardware Detection:'); print('-' * 50); cuda_available = torch.cuda.is_available(); print(f'CUDA Available: {cuda_available}'); if cuda_available: print(f'CUDA Device: {torch.cuda.get_device_name(0)}'); rocm_available = hasattr(torch.version, 'hip') and torch.version.hip is not None; print(f'ROCm Available: {rocm_available}'); mkl_available = torch.backends.mkl.is_available(); print(f'MKL Available: {mkl_available}'); if cuda_available: device = 'cuda'; elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = 'mps'; else: device = 'cpu'; print(f'\nUsing device: {device}'); return device; if __name__ == '__main__': device = check_hardware(); print(f'\nReady for training on {device}!')"
echo Starting training...

:: Set environment variables for training (can be modified as needed)
set EPOCHS=1
set BATCH_SIZE=32
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
