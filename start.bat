@echo off
REM Memory Factory Studio - double-click to start (Windows)
cd /d %~dp0

where python >nul 2>nul || goto :nopython
echo Using:
python --version || goto :nopython

if not exist .venv (
    echo.
    echo [1/3] First run: creating Python environment...
    python -m venv .venv || goto :nopython
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
) else (
    call .venv\Scripts\activate
)

echo.
echo [2/3] Checking packages. The first time, or after an update, this downloads
echo       a few hundred MB and can take 5-15 minutes. Please wait...
echo.
pip install -r requirements.txt --disable-pip-version-check || goto :fail

echo.
echo [3/3] Starting Memory Factory Studio...
echo       Your browser opens at http://127.0.0.1:7860
echo       Keep this window open while you use the app. Close it to stop the app.
echo.
python app.py %*
echo.
echo The app stopped. If you see an error above, copy it and send it for help.
pause
goto :eof

:nopython
echo.
echo Python was not found. Install Python 3.11 or 3.12 from python.org
echo and tick "Add python.exe to PATH" during installation.
pause
goto :eof

:fail
echo.
echo Installing packages failed. Check your internet connection, then run start.bat again.
echo If it keeps failing, copy the red error text above and send it for help.
pause
