@echo off
REM Memory Factory Studio - double-click to start (Windows)
cd /d %~dp0
if not exist .venv (
    echo First run: creating Python environment...
    python -m venv .venv || goto :nopython
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt || goto :fail
) else (
    call .venv\Scripts\activate
    REM Picks up new packages after a git pull - quick when nothing changed
    pip install -q -r requirements.txt
)
python app.py %*
goto :eof

:nopython
echo Python not found. Install Python 3.11 from python.org and tick "Add python.exe to PATH".
pause
goto :eof

:fail
echo Installing packages failed - check your internet connection and try again.
rmdir /s /q .venv
pause
