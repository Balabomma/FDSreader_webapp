@echo off
REM FDSReader WebApp — One-line installer for Windows
REM Usage: curl -fsSL https://raw.githubusercontent.com/Balabomma/FDSreader_webapp/master/install.bat -o install.bat && install.bat
setlocal

set REPO=https://github.com/Balabomma/FDSreader_webapp.git
set INSTALL_DIR=FDSreader_webapp

echo =========================================
echo   FDSReader WebApp — Installer
echo =========================================

REM Check for Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3 is required but not found.
    echo Install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

python --version

REM Check for git
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: git is required but not found.
    echo Install Git from https://git-scm.com/download/win
    exit /b 1
)

REM Clone or update
if exist "%INSTALL_DIR%" (
    echo Directory '%INSTALL_DIR%' already exists. Pulling latest...
    cd "%INSTALL_DIR%"
    git pull
) else (
    echo Cloning repository...
    git clone %REPO% %INSTALL_DIR%
    cd "%INSTALL_DIR%"
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate and install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt

echo.
echo =========================================
echo   Installation complete!
echo =========================================
echo.
echo To run the app:
echo   cd %INSTALL_DIR%
echo   venv\Scripts\activate.bat
echo   python app.py
echo.
echo Then open http://localhost:5000 in your browser.
