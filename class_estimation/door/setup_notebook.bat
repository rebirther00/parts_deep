@echo off
chcp 65001 >nul 2>&1
title Door Dataset Builder - Setup

echo ============================================================
echo   Door Dataset Builder - 환경 설정
echo ============================================================
echo.

REM Python 확인
echo [1/3] Python 설치 확인...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [오류] Python이 설치되어 있지 않거나 PATH에 등록되지 않았습니다.
    echo   1. https://www.python.org/downloads/ 에서 Python 3.10 이상 다운로드
    echo   2. 설치 시 "Add Python to PATH" 체크 필수
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo   %%i 확인됨
echo.

REM ZED SDK Python API 확인
echo [2/3] ZED SDK Python API 확인...
python -c "import pyzed.sl" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [경고] ZED SDK Python API (pyzed)가 설치되어 있지 않습니다.
    echo.
    echo   ZED SDK 설치 후 다음 명령을 실행하세요:
    echo     python "C:\Program Files (x86)\ZED SDK\get_python_api.py"
    echo.
    echo   ZED SDK가 설치되어 있지 않다면:
    echo     https://www.stereolabs.com/developers/release 에서 다운로드
    echo.
    echo   * ZED SDK 없이도 OpenCV 웹캠 모드로 RGB 캡처는 가능합니다.
    echo.
) else (
    echo   pyzed 확인됨
    echo.
)

REM pip 패키지 설치
echo [3/3] Python 패키지 설치...
pip install -r requirements_notebook.txt
if errorlevel 1 (
    echo.
    echo [오류] 패키지 설치 실패. pip가 정상 동작하는지 확인하세요.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   설정 완료! run_notebook.bat 을 실행하세요.
echo ============================================================
echo.
pause
