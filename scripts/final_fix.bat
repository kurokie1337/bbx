@echo off
echo ========================================
echo BBX Console - Final Fix
echo ========================================
echo.

cd /d "%~dp0.."
cd bbx-console

echo Rebuilding backend with all fixes...
docker-compose build backend

echo.
echo Restarting all services...
docker-compose up -d

echo.
echo Waiting 20 seconds for full startup...
timeout /t 20

echo.
echo Testing health...
curl http://localhost:8000/api/health

echo.
echo ========================================
echo Status:
echo ========================================
docker-compose ps

echo.
echo ========================================
echo Backend Logs:
echo ========================================
docker-compose logs --tail=50 backend

echo.
echo ========================================  
echo If backend is running, open:
echo   Frontend: http://localhost:3000
echo   API Docs: http://localhost:8000/docs
echo ========================================
echo.
pause
