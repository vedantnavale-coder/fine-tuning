@echo off
echo ========================================
echo  Local AI Training Platform
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo.
echo Starting backend server...
echo.
echo Backend will run on: http://localhost:8000
echo Open this URL in your browser to access the UI
echo.
echo Press Ctrl+C to stop the server
echo.

cd backend
python app.py
