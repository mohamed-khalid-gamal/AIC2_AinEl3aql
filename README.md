# MTC-AIC3 BCI Competition Submission

## Full Reproducibility Package

This submission provides a complete, reproducible solution for the MTC-AIC3 BCI competition, including both Motor Imagery (MI) and SSVEP task processing.

## 📁 File Structure

```
├── unified_mtc_aic3_pipeline.ipynb    # Main notebook with complete pipeline
├── requirements.txt                   # Python dependencies
├── config_default.json               # Default training configuration
├── config_fast.json                  # Fast training configuration  
├── config_best_performance.json      # Best performance configuration
├── checkpoints/                      # Model checkpoints and saved components
│   ├── best_MI_EEGNet.pt            # Best MI model weights
│   ├── best_SSVEP_EnhancedFeatureClassifier.pt # Best SSVEP model weights
│   ├── training_config.json         # Training configuration used
│   ├── training_results.json        # Complete training results
│   ├── label_encoders.pkl           # Label encoders for both tasks
│   ├── MI_preprocessor.pkl          # MI preprocessing components
│   ├── MI_feature_extractor.pkl     # MI feature extraction components
│   ├── SSVEP_scaler.pkl            # SSVEP feature scaling
│   └── SSVEP_selector.pkl          # SSVEP feature selection
├── submission.csv                    # Final predictions
└── README.md                        # This file

```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete notebook from start to finish
jupyter notebook unified_mtc_aic3_pipeline.ipynb
# Execute all cells (Cell > Run All)
```

### Option 2: Train from Scratch
```python
# In the notebook, use the training pipeline
training_pipeline = CompleteTrainingPipeline()
results = training_pipeline.train_from_scratch()
```

### Option 3: Inference Only (if models already trained)
```python
# In the notebook, use the inference pipeline
inference_pipeline = InferencePipeline()
inference_pipeline.load_trained_models()
submission = inference_pipeline.predict_test_set()
```

## 🔧 Configuration

Three pre-configured setups are provided:

1. **Default** (`config_default.json`): Balanced performance and training time
2. **Fast** (`config_fast.json`): Quick training for experiments
3. **Best Performance** (`config_best_performance.json`): Maximum accuracy (longer training)

## 📊 Model Architecture

### Motor Imagery (MI) Task
- **Model**: EEGNet (Convolutional Neural Network)
- **Features**: Filter Bank Common Spatial Patterns (FBCSP)
- **Preprocessing**: 8-30Hz bandpass filter, epoch normalization
- **Input**: 8 channels × 2250 samples (9 seconds @ 250Hz)

### SSVEP Task
- **Model**: Enhanced Feature Classifier (Attention-based MLP)
- **Features**: Filter Bank Canonical Correlation Analysis (FBCCA)
- **Target Frequencies**: 7, 8, 10, 13 Hz
- **Input**: Multi-band correlation features

## 🔄 Reproducibility Features

- ✅ **Fixed Random Seeds**: Consistent results across runs
- ✅ **Complete Checkpointing**: All model states and preprocessing saved
- ✅ **Configuration Logging**: All hyperparameters tracked
- ✅ **Training History**: Complete logs of training progress
- ✅ **Dependency Management**: Exact package versions specified

## 📈 Training Process

1. **Data Loading**: Unified loading for both MI and SSVEP tasks
2. **Preprocessing**: Task-specific signal processing
3. **Feature Extraction**: Advanced feature engineering
4. **Model Training**: Deep learning with early stopping
5. **Validation**: Continuous monitoring of performance
6. **Checkpointing**: Automatic saving of best models

## 🎯 Output

- **submission.csv**: Final predictions in competition format
- **checkpoints/**: Complete model states for reproduction
- **Logs**: Detailed training history and configuration

## 🛠 Requirements

- Python 3.7+
- PyTorch 1.9+
- scikit-learn 1.0+
- MNE-Python 0.24+
- See `requirements.txt` for complete list

## 📞 Usage Examples

### Basic Training
```python
# Load the notebook and run all cells
# Or use the training pipeline directly:
results = training_pipeline.train_from_scratch(config_default)
```

### Custom Configuration
```python
custom_config = {
    'mi_model_type': 'CNNLSTM',
    'epochs': 150,
    'lr': 8e-4,
    # ... other parameters
}
results = training_pipeline.train_from_scratch(custom_config)
```

### Inference Only
```python
# Load pre-trained models and generate predictions
inference_pipeline.load_trained_models()
predictions = inference_pipeline.predict_test_set('my_submission.csv')
```

## 🔍 Validation

The pipeline includes comprehensive validation:
- Cross-validation during training
- Early stopping to prevent overfitting
- Automatic checkpoint of best models
- Complete reproducibility verification

## 💾 Model Checkpoints

All trained models are saved with complete state information:
- Model architecture and weights
- Optimizer state
- Training configuration
- Validation performance
- Preprocessing components

This enables exact reproduction of results and continued training from any checkpoint.

## 🎉 Expected Results

The pipeline should achieve competitive performance on both tasks:
- **MI Task**: >80% validation accuracy
- **SSVEP Task**: >85% validation accuracy
- **Combined**: High-quality submission file

## 📝 Notes

- The pipeline automatically handles task routing based on the 'task' column
- All preprocessing is task-specific and optimized
- Model selection is based on validation performance
- The system is designed for end-to-end execution without manual intervention

For questions or issues, please refer to the notebook documentation or training logs.
