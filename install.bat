@echo off
echo ========================================
echo  Installing Local AI Training Platform
echo ========================================
echo.

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing dependencies...
echo This may take 10-15 minutes...
pip install -r requirements.txt

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo To start the platform:
echo   1. Run: start.bat
echo   2. Open browser: http://localhost:8000
echo.
pause
