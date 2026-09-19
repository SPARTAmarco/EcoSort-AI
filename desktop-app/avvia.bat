@echo off
setlocal EnableExtensions
chcp 65001 >nul
title EcoSort AI - Avvio app desktop

:: ============================================================================
::  EcoSort AI - avvio con un doppio clic
::
::    avvia.bat              prepara tutto (solo la prima volta) e apre l'app
::    avvia.bat test         apre il banco di prova nel browser (webcam o foto)
::    avvia.bat ricompila    forza la ricompilazione del jar e poi apre l'app
::
::  Prima volta: scarica i modelli dalla Release v1.0.0, crea .venv con
::  TensorFlow, compila il jar con Maven. Dalla seconda volta parte subito.
:: ============================================================================

cd /d "%~dp0"
set "RELEASE=https://github.com/SPARTAmarco/EcoSort-AI/releases/download/v1.0.0"
set "JAR=target\ecosort-ai-1.0.0.jar"
set "VPY=%CD%\.venv\Scripts\python.exe"
set "MODO=%~1"

echo.
echo  ============================================
echo   EcoSort AI - app desktop
echo  ============================================
echo.

:: ---------------------------------------------------------------- 1. modelli
call :scarica newbest_model.keras "69 MB" || goto :errore
call :scarica rifiuti.tflite "8 MB" || goto :errore
call :scarica config.json "1 KB" || goto :errore

:: ---------------------------------------------------------- 2. Python + .venv
if exist "%VPY%" if exist ".venv\.pronto" goto :python_ok

echo [..] Creo l'ambiente Python in .venv  (solo la prima volta, 3-5 minuti)
set "PYBASE="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined PYBASE (
        py -%%V -c "import sys" >nul 2>&1 && set "PYBASE=py -%%V"
    )
)
if not defined PYBASE (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PYBASE=python"
)
if not defined PYBASE (
    echo [ERR] Serve Python 3.10 - 3.13 ^(TensorFlow non supporta ancora versioni piu nuove^).
    echo       Scaricalo da https://www.python.org/downloads/  e rilancia avvia.bat
    goto :errore
)
echo [OK] Python base: %PYBASE%
if not exist "%VPY%" %PYBASE% -m venv .venv || goto :errore
"%VPY%" -m pip install --upgrade pip --quiet || goto :errore
"%VPY%" -m pip install -r server\requirements.txt || goto :errore
echo ok> ".venv\.pronto"

:python_ok
echo [OK] Ambiente Python pronto (.venv)
set "ECOSORT_PYTHON=%VPY%"

if /i "%MODO%"=="test" goto :test

:: ------------------------------------------------------------ 3. Java + jar
where java >nul 2>&1 || (
    echo [ERR] Java non trovato. Serve il JDK 21: https://adoptium.net/
    goto :errore
)
if /i "%MODO%"=="ricompila" if exist "%JAR%" del "%JAR%"
if exist "%JAR%" goto :jar_ok

echo [..] Compilo l'app con Maven (solo la prima volta)
set "MVN="
where mvn >nul 2>&1 && set "MVN=mvn"
if not defined MVN if exist "C:\Program Files\Maven\bin\mvn.cmd" set "MVN=C:\Program Files\Maven\bin\mvn.cmd"
if not defined MVN if exist "C:\tools\maven\bin\mvn.cmd" set "MVN=C:\tools\maven\bin\mvn.cmd"
if not defined MVN (
    echo [ERR] Maven non trovato. Installa Apache Maven e aggiungilo al PATH.
    goto :errore
)
call "%MVN%" -q -B clean package -DskipTests || goto :errore

:jar_ok
echo [OK] %JAR%

:: ------------------------------------------------------------------ 4. avvio
echo.
echo [..] Avvio EcoSort AI. Il primo caricamento di TensorFlow richiede qualche secondo.
echo      Log del server Python: %CD%\ecosort-server.log
echo.
java -jar "%JAR%"
goto :fine

:: ---------------------------------------------------- banco di prova (test)
:test
echo [..] Banco di prova: si apre il browser, usa la webcam o trascina una foto
echo      Stesso modello, stesso preprocessing e stessa regola del Raspberry.
echo.
"%VPY%" "..\tools\prova_pc.py" --modello "server\rifiuti.tflite" --config "server\config.json"
goto :fine

:: ------------------------------------------------------------- sottoroutine
:scarica
if exist "server\%~1" (
    echo [OK] server\%~1
    exit /b 0
)
echo [..] Scarico %~1 ^(%~2^) dalla Release v1.0.0
curl -L --fail --progress-bar -o "server\%~1.part" "%RELEASE%/%~1" || (
    if exist "server\%~1.part" del "server\%~1.part"
    echo [ERR] Download di %~1 fallito. Controlla la connessione.
    exit /b 1
)
move /y "server\%~1.part" "server\%~1" >nul
echo [OK] server\%~1
exit /b 0

:errore
echo.
echo [ERR] Avvio interrotto: leggi il messaggio qui sopra.
pause
exit /b 1

:fine
endlocal
