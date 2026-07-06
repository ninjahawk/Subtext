@echo off
title Subtext
cd /d "%~dp0"
echo Starting Subtext ... (model + lens take ~1 min to load)
start "" /min cmd /c "timeout /t 20 >nul & start http://localhost:8765"
python -u -X utf8 server.py
pause
