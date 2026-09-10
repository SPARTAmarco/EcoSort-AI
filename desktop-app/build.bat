@echo off
chcp 65001 >nul
echo ============================================
echo  EcoSort AI - Maven Build
echo ============================================
cd /d "%~dp0"
echo Cartella: %CD%

:: Cerca mvn nel PATH
where mvn >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [OK] Maven trovato nel PATH
    mvn clean package -e 2>&1
) else (
    echo [ERR] mvn non trovato nel PATH!
    echo Provo con Maven embedded in strumenti comuni...

    :: Prova alcuni percorsi comuni
    if exist "C:\Program Files\Maven\bin\mvn.cmd" (
        "C:\Program Files\Maven\bin\mvn.cmd" clean package -e 2>&1
    ) else if exist "C:\tools\maven\bin\mvn.cmd" (
        "C:\tools\maven\bin\mvn.cmd" clean package -e 2>&1
    ) else (
        echo Maven non trovato. Installa Apache Maven e aggiungi al PATH.
        pause
        exit /b 1
    )
)

echo.
echo ============================================
echo  Build completata
echo ============================================
pause
