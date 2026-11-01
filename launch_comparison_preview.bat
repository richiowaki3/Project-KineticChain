@echo off
chcp 65001 > nul
cd /d "%~dp0onomato-audio-analyzer"
echo [OnomaDict Preview] Launching Before vs After Comparison Studio...
python -m streamlit run src/dict_comparison_preview.py --browser.gatherUsageStats false
pause
