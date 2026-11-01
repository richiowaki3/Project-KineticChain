@echo off
chcp 65001 > nul
echo ============================================================================
echo [Demo 4] Launching Audio Resynthesizer Web App (Streamlit)...
echo ============================================================================
"C:\Users\antigravity\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run demo4_audio_resynthesizer_app.py --server.port 8546 --server.headless false
pause
