@echo off
title NATIONAL MARINE DEBRIS & GHOST NET INTELLIGENCE PLATFORM (NM-DGIP)
echo ===============================================================================
echo   NATIONAL MARINE DEBRIS & GHOST NET INTELLIGENCE PLATFORM (NM-DGIP)
echo   DIRECTORATE GENERAL OF HYDROGRAPHIC SURVEYS ? ACOUSTIC INTELLIGENCE DIVISION
echo ===============================================================================
echo.
echo [1/3] Starting FastAPI Neural Inference Backend on Port 8000...
start "NM-DGIP FastAPI Backend" cmd /k "cd /d D:\sih1 && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/3] Starting React.js GIS & Analysis Frontend on Port 5173...
start "NM-DGIP React Frontend" cmd /k "cd /d D:\sih1\frontend && npm run dev"

echo [3/3] Waiting for servers to initialize...
timeout /t 3 /nobreak >nul

echo Opening browser at http://localhost:5173...
start http://localhost:5173

echo.
echo ===============================================================================
echo   SYSTEM ONLINE:
echo   - React Frontend: http://localhost:5173
echo   - FastAPI Backend API: http://localhost:8000
echo   - API Swagger Docs: http://localhost:8000/docs
echo ===============================================================================
pause
