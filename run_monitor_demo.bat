@echo off
REM 5단계 시연: 받는 창(monitor)을 먼저 띄우고, 3초 뒤 보내는 창(signal_generator)을 10배속으로 띄운다.
chcp 65001 >nul
set PY=C:\Users\test\AppData\Local\Programs\Python\Python314\python.exe
if not exist "%PY%" set PY=python
cd /d %~dp0
start "monitor - 감지와 예측" cmd /k ""%PY%" monitor.py"
timeout /t 3 >nul
start "signal_generator - 센서 흉내 (정답이 보이는 창)" cmd /k ""%PY%" signal_generator.py --speed 10 --max-startups 215"
