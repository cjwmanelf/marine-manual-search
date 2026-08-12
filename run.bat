@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo.
echo ============================================================
echo   선박기기 매뉴얼 검색 앱을 시작합니다.
echo   라이브러리와 AI 모델을 불러오는 중이라 시간이 걸립니다.
echo   * 처음 실행은 모델 다운로드 때문에 수 분 걸릴 수 있어요.
echo   * 화면에 http://127.0.0.1:7860 주소가 뜨면 준비 완료입니다.
echo   잠시만 기다려 주세요...
echo ============================================================
echo.
"C:\Program Files\Python314\python.exe" app.py
pause
