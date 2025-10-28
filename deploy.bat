@echo off
echo ========================================
echo   Medical RAG System Deployment
echo ========================================
echo.

echo [1/4] Stopping existing servers...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/4] Starting Flask server...
start "Flask Server" /MIN cmd /c "cd /d D:\hack acure\Dataset && python app.py"
echo     Flask server starting...

echo [3/4] Waiting for Flask to initialize...
timeout /t 25 /nobreak >nul

echo [4/4] Starting ngrok tunnel...
start "Ngrok Tunnel" /MIN cmd /c "cd /d D:\hack acure && ngrok.exe http 5000"
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo   DEPLOYMENT STARTED!
echo ========================================
echo.
echo Check the minimized windows for Flask and Ngrok
echo.
echo To get your public URL:
echo   1. Open http://127.0.0.1:4040 in your browser
echo   2. Copy the https URL shown
echo.
echo API Endpoint: YOUR_NGROK_URL/api/ask
echo.
echo Test locally: python test_api.py
echo.
pause
