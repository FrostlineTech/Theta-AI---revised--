@echo off
echo Theta AI - Web Interface
echo ====================================
echo.
echo This script will start the Theta AI web interface
echo Started at: %date% %time%
echo.

REM Set environment variables
set PYTHONPATH=.

REM Run the web interface
python src/web_interface.py

echo.
echo Web interface closed at: %date% %time%
echo.

pause
