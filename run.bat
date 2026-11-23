@echo off
chcp 65001 >nul
title Point-SAM 3D Segmentation - Web Demo

:: Evita problemi di percorso di rete (UNC) creando una mappatura temporanea se necessario
pushd "%~dp0" 2>nul

:: ==============================================================================
::                         CONFIGURAZIONE PARAMETRI DEMO
:: ==============================================================================
:: Modifica qui i parametri per configurare l'esecuzione:

:: 1. Nome del file della nuvola (collocato in demo/static/models/)
::    Esempi: temp.ply, temp2.ply, temp3.ply, scene.ply, rhino.ply
set POINTCLOUD=temp3.ply

:: 2. Numero massimo di punti consentiti (sfoltimento se superato)
::    - 50000: consigliato per fluidita' web e velocita' di calcolo
::    - 100000: per nuvole molto grandi con maggior dettaglio
::    - 0 oppure -1: disabilita lo sfoltimento
set MAX_POINTS=100000

:: 3. Dimensione del voxel per downsampling geometrico preliminare (opzionale)
::    - Lasciare vuoto per non usarlo
::    - Esempio: 0.02
set VOXEL_SIZE=

:: 4. Host e Porta del server locale
set HOST=localhost
set PORT=5000

:: 5. Percorso del checkpoint pesi del modello
set CKPT_PATH=./pretrained/model.safetensors

:: 6. Configurazione Hydra del modello (large, small, etc.)
set CONFIG=large

:: 7. Aprire automaticamente il browser all'avvio? (SI / NO)
set APRI_BROWSER=SI

:: ==============================================================================
::                  ESECUZIONE IN AMBIENTE WSL UBUNTU
:: ==============================================================================

set PROJECT_DIR=/home/matteo/PointSAM/PointSAM
set CONDA_ACTIVATE=/home/cnr/miniforge3/bin/activate

:: Costruzione argomenti riga di comando
set ARGS=--host %HOST% --port %PORT% --pointcloud %POINTCLOUD% --config %CONFIG% --ckpt_path %CKPT_PATH% --max_points %MAX_POINTS%

if not "%VOXEL_SIZE%"=="" (
    set ARGS=%ARGS% --voxel_size %VOXEL_SIZE%
)

cls
echo ==============================================================================
echo                      AVVIO DEMO POINT-SAM (WSL)
echo ==============================================================================
echo  - Nuvola selezionata : %POINTCLOUD%
echo  - Limite max punti   : %MAX_POINTS%
if not "%VOXEL_SIZE%"=="" echo  - Voxel size         : %VOXEL_SIZE%
echo  - Indirizzo Web      : http://%HOST%:%PORT%
echo ==============================================================================
echo.
echo Avvio del server backend in corso...
echo Premi CTRL+C in questa finestra per arrestare il server.
echo.

if /I "%APRI_BROWSER%"=="SI" (
    start http://%HOST%:%PORT%
)

wsl.exe -d Ubuntu --cd %PROJECT_DIR% bash -c "source %CONDA_ACTIVATE% pointsam && python demo/app.py %ARGS%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ==============================================================================
    echo [ERRORE] Il server si e' interrotto con codice di errore: %ERRORLEVEL%
    echo ==============================================================================
    pause
)

popd 2>nul
