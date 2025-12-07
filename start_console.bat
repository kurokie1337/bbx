@echo off
echo [BBX] Starting BBX Console...

cd bbx-console

if not exist .env (
    echo [BBX] Creating .env from example...
    copy .env.example .env
)

echo [BBX] Building and starting containers...
docker compose up -d --build

echo [BBX] Console is running at http://localhost:3000
pause
