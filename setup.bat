@echo off

:: Check for Python in the system path
where python >nul 2>&1
if %errorlevel% equ 0 (
    set PY_EXECUTABLE=python
) else (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        set PY_EXECUTABLE=python3
    ) else (
        echo Error: Python not found in system path
        exit /b 1
    )
)

:: Check for venv module in the Python installation
%PY_EXECUTABLE% -c "import venv" >nul 2>&1
if %errorlevel% equ 0 (
    set VENV_CMD=%PY_EXECUTABLE% -m venv
) else (
    echo Error: venv module not found in Python installation
    exit /b 1
)

:: Create a new virtual environment
%VENV_CMD% instruct-grid
call .\instruct-grid\Scripts\activate.bat

:: Install the required packages
pip install -r requirements.txt

:: Run the setup script
%PY_EXECUTABLE% app.py
