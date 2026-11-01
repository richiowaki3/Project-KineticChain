@echo off
chcp 65001 > nul
cd /d "%~dp0onomato-audio-analyzer"
echo [Onomato Audio Analyzer] Launching Streamlit Studio...
python -m streamlit run src/analyzer_ui.py --browser.gatherUsageStats false
pause
