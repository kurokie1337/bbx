@echo off
cd /d "%~dp0.."
cd bbx-console

echo ========================================
echo BBX Console - Live Logs
echo ========================================
echo.
echo Watching backend logs in real-time...
echo Open UI at http://localhost:3000 and watch requests here
echo Press Ctrl+C to stop
echo.

docker-compose logs -f backend
