@echo off
cd /d "%~dp0"
python run_reverse_labeller.py
if errorlevel 1 pause
