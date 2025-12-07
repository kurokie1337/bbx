@echo off
echo ========================================
echo BBX Console - Rebuild Backend (Fixed)
echo ========================================
echo.

cd /d "%~dp0.."
cd bbx-console

echo Stopping backend...
docker-compose stop backend

echo.
echo Rebuilding backend with CORS fix...
docker-compose build backend

echo.
echo Starting backend...
docker-compose up -d backend

echo.
echo Waiting 10 seconds for startup...
timeout /t 10

echo.
echo Testing health endpoint...
curl http://localhost:8000/api/health

echo.
echo ========================================
echo Status:
echo ========================================
docker-compose ps

echo.
echo ========================================
echo Backend logs:
echo ========================================
docker-compose logs --tail=20 backend

echo.
pause
