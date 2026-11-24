@echo off
chcp 65001 >nul
title Point-SAM - Pipeline Automatica (Modello 1 + Modello 2)

pushd "%~dp0" 2>nul

:: ==============================================================================
::                       CONFIGURAZIONE PARAMETRI PIPELINE
:: ==============================================================================

:: 1. Percorso COMPLETO al file della nuvola (.las o .ply)
::    Esempi:
::      set INPUT=demo\static\models\scene.ply
::      set INPUT=C:\Utenti\Matteo\Documenti\rilievo_bosco.las
set INPUT=demo\static\models\ticino2.las

:: 2. Soglia massima di punti (sfoltimento automatico se superato)
::    - 0     : ADATTIVO AUTOMATICO (consigliato: adatta densita' ed estensione alla scena)
::    - 60000 : limite fisso rapido
::    - 100000: per scene molto estese
set MAX_POINTS=0

:: 3. Voxel size per downsampling geometrico (lascia vuoto per non usarlo)
::    Esempio: 0.02
set VOXEL_SIZE=

:: 4. Risoluzione immagine 2D nadirale (in pixel del lato lungo)
set RISOLUZIONE=1024

:: ==============================================================================

set PROJECT_DIR=/home/matteo/PointSAM/PointSAM
set CONDA_ACTIVATE=/home/cnr/miniforge3/bin/activate

:: Converti il percorso Windows in percorso WSL se necessario
set INPUT_WSL=%INPUT:\=/%
if not "%INPUT_WSL:~1,1%"==":" (
    :: Percorso relativo: prefissa con la dir del progetto
    set INPUT_WSL=%PROJECT_DIR%/%INPUT_WSL%
) else (
    :: Percorso assoluto Windows (es. C:\...) -> /mnt/c/...
    set DRIVE=%INPUT_WSL:~0,1%
    call :tolower %DRIVE% DRIVE_L
    set INPUT_WSL=/mnt/%DRIVE_L%/%INPUT_WSL:~3%
)

set ARGS=%INPUT_WSL% --max_points %MAX_POINTS% --risoluzione %RISOLUZIONE%

if not "%VOXEL_SIZE%"=="" (
    set ARGS=%ARGS% --voxel_size %VOXEL_SIZE%
)

cls
echo ==============================================================================
echo       PIPELINE AUTOMATICA - SAM 2D + Point-SAM 3D
echo ==============================================================================
echo  - File input        : %INPUT%
if not "%VOXEL_SIZE%"=="" echo  - Voxel size        : %VOXEL_SIZE%
echo  - Risoluzione 2D    : %RISOLUZIONE% px
echo ==============================================================================
echo.

wsl.exe -d Ubuntu --cd %PROJECT_DIR% bash -c "source %CONDA_ACTIVATE% pointsam && python pipeline.py %ARGS%"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ==============================================================================
    echo  COMPLETATO. I risultati si trovano in: results\
    echo ==============================================================================
) else (
    echo.
    echo ==============================================================================
    echo  [ERRORE] La pipeline si e' interrotta con codice: %ERRORLEVEL%
    echo ==============================================================================
)

pause
popd 2>nul
goto :eof

:tolower
set "_str=%~1"
for %%A in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    set "_str=!_str:%%A=%%A!"
)
set "%~2=%_str:~0,1%"
goto :eof
