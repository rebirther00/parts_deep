@echo off
chcp 65001 >nul 2>&1
title Door Dataset Builder

echo ============================================================
echo   Door Dataset Builder 시작 중...
echo ============================================================
echo.

REM Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python을 찾을 수 없습니다. setup_notebook.bat을 먼저 실행하세요.
    pause
    exit /b 1
)

echo 서버 시작: http://localhost:5000
echo 브라우저가 자동으로 열립니다.
echo 종료하려면 이 창에서 Ctrl+C 를 누르세요.
echo.

REM 2초 후 브라우저 열기 (백그라운드)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

REM 서버 실행
python 01_capture_dataset_notebook.py

echo.
echo 서버가 종료되었습니다.
pause
