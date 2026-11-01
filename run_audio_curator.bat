@echo off
cd /d "%~dp0"
python run_audio_curator.py
if errorlevel 1 pause
