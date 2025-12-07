@echo off
echo ========================================
echo BBX Console - Debug Backend
echo ========================================
echo.

cd /d "%~dp0.."
cd bbx-console

echo Checking container status...
docker-compose ps

echo.
echo ========================================
echo Backend Logs (last 50 lines):
echo ========================================
docker-compose logs --tail=50 backend

echo.
echo ========================================
echo Testing Backend Health:
echo ========================================
curl http://localhost:8000/api/health

echo.
echo.
echo ========================================
echo Testing Backend Docs:
echo ========================================
curl http://localhost:8000/docs

echo.
pause
