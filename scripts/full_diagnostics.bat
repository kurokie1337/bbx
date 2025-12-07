@echo off
cd /d "%~dp0.."
cd bbx-console

echo ========================================
echo FULL Backend Logs (all entries):
echo ========================================
docker-compose logs backend

echo.
echo ========================================
echo Container Status:
echo ========================================
docker-compose ps backend

echo.
echo ========================================  
echo Inspecting Container:
echo ========================================
docker inspect bbx-console-backend

echo.
pause
