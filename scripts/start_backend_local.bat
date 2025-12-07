@echo off
echo ========================================
echo BBX Console - Local Development Start
echo ========================================
echo.

echo Starting Backend API...
echo.

cd bbx-console\backend

echo Installing dependencies...
pip install -q fastapi uvicorn pydantic pydantic-settings websockets sqlalchemy aiosqlite httpx python-multipart pyyaml python-dotenv

echo.
echo Starting FastAPI server on http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo.

start /B uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

echo.
echo ========================================
echo Backend started!
echo ========================================
echo.
echo API: http://localhost:8000
echo Docs: http://localhost:8000/docs
echo.
echo Now open another terminal and run:
echo   cd bbx-console\frontend
echo   npm install
echo   npm run dev
echo.
pause
