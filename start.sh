#!/bin/bash

echo "========================================"
echo " Local AI Training Platform"
echo "========================================"
echo ""

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found. Please install Python 3.10+"
    exit 1
fi

echo ""
echo "Starting backend server..."
echo ""
echo "Backend will run on: http://localhost:8000"
echo "Open this URL in your browser to access the UI"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd backend
python3 app.py
