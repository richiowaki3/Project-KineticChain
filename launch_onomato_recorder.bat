@echo off
chcp 65001 > nul
cd /d "%~dp0onomato-audio-recorder"
echo [Onomato TTS Rec Studio] Launching TTS Recording Studio...
python -m streamlit run src/app_ui.py --browser.gatherUsageStats false
pause
