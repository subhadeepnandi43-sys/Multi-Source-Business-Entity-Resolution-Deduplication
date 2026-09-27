@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting EntityResolver Localhost Studio
echo URL: http://localhost:8000
echo ========================================================
.\.venv\Scripts\python.exe server.py
pause
