@echo off
REM 1~4단계를 차례로 실행한다. PY = torch가 설치된 파이썬
chcp 65001 >nul
set PY=C:\Users\test\AppData\Local\Programs\Python\Python314\python.exe
if not exist "%PY%" set PY=python
cd /d %~dp0
"%PY%" simulate.py || goto :err
"%PY%" train.py || goto :err
"%PY%" evaluate.py || goto :err
"%PY%" run_realtime.py || goto :err
echo.
echo 완료. reports\test_report.md 와 reports\examples.png 를 열어 보세요.
pause
exit /b 0
:err
echo 오류로 멈춤.
pause
exit /b 1
