@echo off
echo ========================================
echo BBX Console - Rebuild Frontend (API Fix)
echo ========================================
echo.

cd /d "%~dp0.."
cd bbx-console

echo Rebuilding frontend with API URL fix...
docker-compose build frontend

echo.
echo Restarting frontend...
docker-compose up -d frontend

echo.
echo Waiting 10 seconds...
timeout /t 10

echo.
echo Done! Open http://localhost:3000
echo.
pause
