@echo off
title Katheryne Assistant Manager
cls

:check_python
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or newer
    pause
    exit /b 1
)

:check_deps
echo Checking dependencies...
python -c "import tkinter" > nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

:start_gui
echo Starting Katheryne Assistant Manager...
python python/katheryne_gui.py
if errorlevel 1 (
    echo Error starting the application
    pause
)