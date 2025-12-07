@echo off
echo ========================================
echo BBX Backend - Full Logs
echo ========================================
echo.

cd /d "%~dp0.."
cd bbx-console

echo Getting ALL backend logs...
echo.
docker-compose logs backend

echo.
echo ========================================
echo Container Status:
echo ========================================
docker-compose ps backend

echo.
pause
