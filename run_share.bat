@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo.
echo ============================================================
echo   [공유 모드] 선박기기 매뉴얼 검색 앱을 시작합니다.
echo   * 잠시 후 콘솔에 https://xxxx.gradio.live 공개 링크가 나옵니다.
echo   * 그 링크를 복사해 상대에게 전달하세요 (창을 닫으면 링크 끊김).
echo   * 내 화면은 자동으로 브라우저에 열립니다.
echo   [주의] 사내 매뉴얼은 올리지 마세요 (링크를 아는 누구나 접근 가능).
echo ============================================================
echo.
"C:\Program Files\Python314\python.exe" app.py --share --open
pause
