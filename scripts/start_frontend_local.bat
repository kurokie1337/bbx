@echo off
echo ========================================
echo BBX Console - Frontend Start
echo ========================================
echo.

cd bbx-console\frontend

echo Installing dependencies...
call npm install

echo.
echo Starting Vite dev server...
echo.

call npm run dev

pause
