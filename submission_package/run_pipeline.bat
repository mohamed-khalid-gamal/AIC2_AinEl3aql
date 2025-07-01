@echo off
setlocal enabledelayedexpansion

REM EEG Classification Pipeline Execution Script for Windows
REM This script runs the complete pipeline from preprocessing to evaluation

REM Configuration
set DATA_PATH=.\data
set PREPROCESSED_PATH=.\preprocessed_data
set CHECKPOINTS_PATH=.\checkpoints
set RESULTS_PATH=.\results
set SUBMISSION_PATH=.\submission.csv

REM Default parameters
set MI_MODEL=EEGNet
set SSVEP_MODEL=EnhancedFeatureClassifier
set EPOCHS=50
set BATCH_SIZE=32
set LR=0.001
set SKIP_PREPROCESS=false
set SKIP_TRAIN=false
set SKIP_EVALUATE=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--data_path" (
    set DATA_PATH=%~2
    shift /1
    shift /1
    goto parse_args
)
if "%~1"=="--mi_model" (
    set MI_MODEL=%~2
    shift /1
    shift /1
    goto parse_args
)
if "%~1"=="--ssvep_model" (
    set SSVEP_MODEL=%~2
    shift /1
    shift /1
    goto parse_args
)
if "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift /1
    shift /1
    goto parse_args
)
if "%~1"=="--batch_size" (
    set BATCH_SIZE=%~2
    shift /1
    shift /1
    goto parse_args
)
if "%~1"=="--lr" (
    set LR=%~2
    shift /1
    shift /1
    goto parse_args
)
if "%~1"=="--skip_preprocess" (
    set SKIP_PREPROCESS=true
    shift /1
    goto parse_args
)
if "%~1"=="--skip_train" (
    set SKIP_TRAIN=true
    shift /1
    goto parse_args
)
if "%~1"=="--skip_evaluate" (
    set SKIP_EVALUATE=true
    shift /1
    goto parse_args
)
if "%~1"=="--help" (
    echo EEG Classification Pipeline
    echo.
    echo Usage: %0 [OPTIONS]
    echo.
    echo Options:
    echo   --data_path PATH          Path to dataset ^(default: .\data^)
    echo   --mi_model MODEL         MI model type ^(default: EnhancedFeatureClassifier^)
    echo   --ssvep_model MODEL      SSVEP model type ^(default: SSVEPFormer^)
    echo   --epochs N               Training epochs ^(default: 50^)
    echo   --batch_size N           Batch size ^(default: 32^)
    echo   --lr FLOAT               Learning rate ^(default: 0.001^)
    echo   --skip_preprocess        Skip preprocessing step
    echo   --skip_train            Skip training step
    echo   --skip_evaluate         Skip evaluation step
    echo   --help                  Show this help message
    echo.
    echo Example:
    echo   %0 --data_path C:\path\to\data --epochs 100 --mi_model EEGNet
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:end_parse

REM Create directories
if not exist "%PREPROCESSED_PATH%" mkdir "%PREPROCESSED_PATH%"
if not exist "%CHECKPOINTS_PATH%" mkdir "%CHECKPOINTS_PATH%"
if not exist "%RESULTS_PATH%" mkdir "%RESULTS_PATH%"

echo ==================================================
echo EEG Classification Pipeline
echo ==================================================
echo Data path: %DATA_PATH%
echo MI model: %MI_MODEL%
echo SSVEP model: %SSVEP_MODEL%
echo Epochs: %EPOCHS%
echo Batch size: %BATCH_SIZE%
echo Learning rate: %LR%
echo ==================================================

REM Step 1: Preprocessing
if "%SKIP_PREPROCESS%"=="true" (
    echo.
    echo ⏭️  Skipping preprocessing step
) else (
    echo.
    echo Step 1: Data Preprocessing
    echo ==========================
    python preprocess.py --data_path "%DATA_PATH%" --output_path "%PREPROCESSED_PATH%" --mi_feature_type FBCSP --ssvep_feature_type FBCCA --scale_features --select_features 500
    
    if !errorlevel! neq 0 (
        echo ❌ Preprocessing failed!
        exit /b 1
    )
    echo ✅ Preprocessing completed successfully
)

REM Step 2: Training
if "%SKIP_TRAIN%"=="true" (
    echo.
    echo ⏭️  Skipping training step
) else (
    echo.
    echo Step 2: Model Training
    echo =====================
    python train.py --data_path "%PREPROCESSED_PATH%" --save_path "%CHECKPOINTS_PATH%" --task both --mi_model "%MI_MODEL%" --ssvep_model "%SSVEP_MODEL%" --epochs %EPOCHS% --batch_size %BATCH_SIZE% --lr %LR% --patience 15
    
    if !errorlevel! neq 0 (
        echo ❌ Training failed!
        exit /b 1
    )
    echo ✅ Training completed successfully
)

REM Step 3: Inference
echo.
echo Step 3: Generating Predictions
echo ==============================
python inference.py --data_path "%PREPROCESSED_PATH%" --model_path "%CHECKPOINTS_PATH%" --output_path "%SUBMISSION_PATH%" --task both --batch_size %BATCH_SIZE%

if !errorlevel! neq 0 (
    echo ❌ Inference failed!
    exit /b 1
)
echo ✅ Predictions generated successfully

REM Step 4: Evaluation
if "%SKIP_EVALUATE%"=="true" (
    echo.
    echo ⏭️  Skipping evaluation step
) else (
    echo.
    echo Step 4: Model Evaluation
    echo =======================
    python evaluate.py --data_path "%PREPROCESSED_PATH%" --model_path "%CHECKPOINTS_PATH%" --output_path "%RESULTS_PATH%" --task both --cv_folds 5
    
    if !errorlevel! neq 0 (
        echo ❌ Evaluation failed!
        exit /b 1
    )
    echo ✅ Evaluation completed successfully
)

echo.
echo ==================================================
echo 🎉 Pipeline completed successfully!
echo ==================================================
echo Generated files:
echo   📄 Submission: %SUBMISSION_PATH%
echo   📁 Models: %CHECKPOINTS_PATH%
echo   📊 Results: %RESULTS_PATH%
echo   🗃️  Preprocessed: %PREPROCESSED_PATH%
echo.
echo Next steps:
echo   1. Review evaluation reports in %RESULTS_PATH%
echo   2. Submit %SUBMISSION_PATH% to competition
echo   3. Fine-tune hyperparameters if needed
echo ==================================================

endlocal
