@echo off
chcp 65001 > nul
cd /d "%~dp0"
title OnomaPoet Synth

if exist "C:\Users\antigravity\AppData\Local\Programs\Python\Python310\python.exe" (
    "C:\Users\antigravity\AppData\Local\Programs\Python\Python310\python.exe" run_onoma_poet.py
) else (
    python run_onoma_poet.py
)

pause
