@echo off
echo ========================================
echo BBX Console - Quick Start
echo ========================================
echo.

cd /d "%~dp0.."
cd bbx-console

echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running!
    echo Please install Docker Desktop and start it.
    pause
    exit /b 1
)

echo.
echo Starting BBX Console with Docker Compose...
echo This will start:
echo   - PostgreSQL database
echo   - Redis cache
echo   - Backend API (port 8000)
echo   - Frontend UI (port 3000)
echo.

docker-compose up -d

echo.
echo ========================================
echo BBX Console is starting!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo To view logs: docker-compose logs -f
echo To stop: docker-compose down
echo.
pause
