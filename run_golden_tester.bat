@echo off
chcp 65001 > nul
cd /d "%~dp0"
python run_golden_tester.py
pause
