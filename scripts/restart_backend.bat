@echo off
echo ========================================
echo BBX Console - Restart Backend
echo ========================================
echo.

cd /d "%~dp0.."
cd bbx-console

echo Restarting backend container...
docker-compose restart backend

echo.
echo Waiting for backend to start...
timeout /t 10

echo.
echo Checking status...
docker-compose ps backend

echo.
echo Testing health endpoint...
curl http://localhost:8000/api/health

echo.
pause
