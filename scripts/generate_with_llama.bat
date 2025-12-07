@echo off
echo ========================================
echo BBX - Generate Crazy App with Llama
echo ========================================
echo.

echo This will:
echo 1. Check if llama-cpp-python is installed
echo 2. Check if Qwen model is downloaded
echo 3. Generate a crazy workflow using local LLM
echo.

python generate_crazy_app.py

echo.
echo ========================================
echo Generation Complete!
echo ========================================
echo.
echo Check examples/crazy_app.bbx for the generated workflow
echo.
pause
