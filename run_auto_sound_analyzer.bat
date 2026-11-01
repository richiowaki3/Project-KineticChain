@echo off
cd /d "%~dp0"
python run_crawler_interactive.py
if errorlevel 1 pause
