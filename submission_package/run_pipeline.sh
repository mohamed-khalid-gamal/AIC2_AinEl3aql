#!/bin/bash

# EEG Classification Pipeline Execution Script
# This script runs the complete pipeline from preprocessing to evaluation

set -e  # Exit on any error

# Configuration
DATA_PATH="./data"
PREPROCESSED_PATH="./preprocessed_data"
CHECKPOINTS_PATH="./checkpoints"
RESULTS_PATH="./results"
SUBMISSION_PATH="./submission.csv"

# Default parameters
MI_MODEL="EEGNet"
SSVEP_MODEL="EnhancedFeatureClassifier"
EPOCHS=50
BATCH_SIZE=32
LR=0.001

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --mi_model)
            MI_MODEL="$2"
            shift 2
            ;;
        --ssvep_model)
            SSVEP_MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --skip_preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip_train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip_evaluate)
            SKIP_EVALUATE=true
            shift
            ;;
        --help)
            echo "EEG Classification Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_path PATH          Path to dataset (default: ./data)"
            echo "  --mi_model MODEL         MI model type (default: EnhancedFeatureClassifier)"
            echo "  --ssvep_model MODEL      SSVEP model type (default: SSVEPFormer)"
            echo "  --epochs N               Training epochs (default: 50)"
            echo "  --batch_size N           Batch size (default: 32)"
            echo "  --lr FLOAT               Learning rate (default: 0.001)"
            echo "  --skip_preprocess        Skip preprocessing step"
            echo "  --skip_train            Skip training step"
            echo "  --skip_evaluate         Skip evaluation step"
            echo "  --help                  Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --data_path /path/to/data --epochs 100 --mi_model EEGNet"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$PREPROCESSED_PATH" "$CHECKPOINTS_PATH" "$RESULTS_PATH"

echo "=================================================="
echo "EEG Classification Pipeline"
echo "=================================================="
echo "Data path: $DATA_PATH"
echo "MI model: $MI_MODEL"
echo "SSVEP model: $SSVEP_MODEL"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "=================================================="

# Step 1: Preprocessing
if [ "$SKIP_PREPROCESS" != true ]; then
    echo ""
    echo "Step 1: Data Preprocessing"
    echo "=========================="
    python preprocess.py \
        --data_path "$DATA_PATH" \
        --output_path "$PREPROCESSED_PATH" \
        --mi_feature_type FBCSP \
        --ssvep_feature_type FBCCA \
        --scale_features \
        --select_features 500
    
    if [ $? -ne 0 ]; then
        echo "❌ Preprocessing failed!"
        exit 1
    fi
    echo "✅ Preprocessing completed successfully"
else
    echo "⏭️  Skipping preprocessing step"
fi

# Step 2: Training
if [ "$SKIP_TRAIN" != true ]; then
    echo ""
    echo "Step 2: Model Training"
    echo "====================="
    python train.py \
        --data_path "$PREPROCESSED_PATH" \
        --save_path "$CHECKPOINTS_PATH" \
        --task both \
        --mi_model "$MI_MODEL" \
        --ssvep_model "$SSVEP_MODEL" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --patience 15
    
    if [ $? -ne 0 ]; then
        echo "❌ Training failed!"
        exit 1
    fi
    echo "✅ Training completed successfully"
else
    echo "⏭️  Skipping training step"
fi

# Step 3: Inference
echo ""
echo "Step 3: Generating Predictions"
echo "=============================="
python inference.py \
    --data_path "$PREPROCESSED_PATH" \
    --model_path "$CHECKPOINTS_PATH" \
    --output_path "$SUBMISSION_PATH" \
    --task both \
    --batch_size "$BATCH_SIZE"

if [ $? -ne 0 ]; then
    echo "❌ Inference failed!"
    exit 1
fi
echo "✅ Predictions generated successfully"

# Step 4: Evaluation
if [ "$SKIP_EVALUATE" != true ]; then
    echo ""
    echo "Step 4: Model Evaluation"
    echo "======================="
    python evaluate.py \
        --data_path "$PREPROCESSED_PATH" \
        --model_path "$CHECKPOINTS_PATH" \
        --output_path "$RESULTS_PATH" \
        --task both \
        --cv_folds 5
    
    if [ $? -ne 0 ]; then
        echo "❌ Evaluation failed!"
        exit 1
    fi
    echo "✅ Evaluation completed successfully"
else
    echo "⏭️  Skipping evaluation step"
fi

echo ""
echo "=================================================="
echo "🎉 Pipeline completed successfully!"
echo "=================================================="
echo "Generated files:"
echo "  📄 Submission: $SUBMISSION_PATH"
echo "  📁 Models: $CHECKPOINTS_PATH"
echo "  📊 Results: $RESULTS_PATH"
echo "  🗃️  Preprocessed: $PREPROCESSED_PATH"
echo ""
echo "Next steps:"
echo "  1. Review evaluation reports in $RESULTS_PATH"
echo "  2. Submit $SUBMISSION_PATH to competition"
echo "  3. Fine-tune hyperparameters if needed"
echo "=================================================="
