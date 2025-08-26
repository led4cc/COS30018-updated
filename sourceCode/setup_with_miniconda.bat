@echo off
setlocal enabledelayedexpansion

REM === Khởi tạo biến ===
set "INSTALLER=%USERPROFILE%\miniconda_installer.exe"
set "INSTALL_DIR=%USERPROFILE%\miniconda3"

REM === Kiểm tra conda ===
where conda >nul 2>nul
if errorlevel 1 (
    echo [INFO] Conda not found. Proceeding to download and install Miniconda...

    echo [INFO] Downloading Miniconda installer to: %INSTALLER%
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%INSTALLER%'"

    echo [INFO] Installing Miniconda silently...
    call "%INSTALLER%" /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%INSTALL_DIR%

    if exist "%INSTALLER%" (
        del "%INSTALLER%"
    )
) else (
    echo [INFO] Conda is already installed.
)

REM === Kích hoạt Conda ===
if exist "%INSTALL_DIR%\Scripts\activate.bat" (
    call "%INSTALL_DIR%\Scripts\activate.bat"
) else (
    echo [ERROR] activate.bat not found. Miniconda installation may have failed.
    pause
    exit /b
)

REM === Tạo environment nếu chưa có ===
call conda info --envs | findstr /C:"myprojectenv" >nul
if errorlevel 1 (
    echo [INFO] Creating environment myprojectenv...
    call conda create -n myprojectenv python=3.10.16 -y
)

REM === Kích hoạt environment ===
call conda activate myprojectenv

REM === Cài đặt requirements ===
if exist requirements.txt (
    echo [INFO] Installing Python packages from requirements.txt...
    pip install -r requirements.txt
) else (
    echo [WARNING] requirements.txt not found. Skipping package installation.
)

REM === Kết thúc ===
echo [SUCCESS] Setup complete. Closing in 5 seconds...
timeout /t 5 >nul
exit
