@echo off
TITLE ElderGuard AI - Privacy-Preserving Edge Fall Detection
echo ==============================================================================
echo                 ELDERGUARD AI - CAREGIVER EDGE STATION
echo      Navonmesh 26 - PS-06: Edge AI-Based Elderly Fall Detection
echo                   In Collaboration with Qualcomm
echo ==============================================================================
echo.
echo Starting ElderGuard AI Dashboard...
py -m streamlit run app.py --server.headless false
pause
