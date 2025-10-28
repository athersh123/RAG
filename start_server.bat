@echo off
echo ========================================
echo Medical RAG Web Server
echo ========================================
echo.
echo Starting Flask server...
echo.
cd /d "%~dp0Dataset"
python app.py
pause
