@echo off
REM DRISTI - Lost Person Detection System
REM Startup Script for Windows

echo.
echo ======================================================================
echo DRISTI - Lost Person Detection System
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check if CCTVS folder exists
if not exist "CCTVS" (
    echo.
    echo WARNING: CCTVS folder not found
    echo Please add your video files to the CCTVS folder
    echo.
)

REM Start the server
echo.
echo Starting DRISTI Backend Server...
echo.
python main.py

pause
