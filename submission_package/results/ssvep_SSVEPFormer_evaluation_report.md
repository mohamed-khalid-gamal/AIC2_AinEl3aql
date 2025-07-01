# SSVEP SSVEPFormer Model Evaluation Report

## Model Information

- **Task**: SSVEP
- **Model Type**: SSVEPFormer
- **Evaluation Date**: 2025-06-30 22:10:07

## Validation Set Results

- **Accuracy**: 0.2800
- **F1 Score (Weighted)**: 0.1225
- **F1 Score (Macro)**: 0.1094
- **Precision (Weighted)**: 0.0784
- **Recall (Weighted)**: 0.2800

### Per-Class Metrics

- **Class Backward**:
  - Precision: 0.2800
  - Recall: 1.0000
  - F1: 0.4375
- **Class Forward**:
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
- **Class Left**:
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000
- **Class Right**:
  - Precision: 0.0000
  - Recall: 0.0000
  - F1: 0.0000

## Training History

- **Final Training Accuracy**: 0.2646
- **Best Validation Accuracy**: 0.2800
- **Final Validation F1**: 0.1225
- **Training Epochs**: 17

## Detailed Classification Report

```
              precision    recall  f1-score   support

    Backward       0.28      1.00      0.44        14
     Forward       0.00      0.00      0.00        12
        Left       0.00      0.00      0.00        14
       Right       0.00      0.00      0.00        10

    accuracy                           0.28        50
   macro avg       0.07      0.25      0.11        50
weighted avg       0.08      0.28      0.12        50

```

## Prediction Confidence Statistics

- **Mean Confidence**: 0.2593
- **Std Confidence**: 0.0000
- **Min Confidence**: 0.2593
- **Max Confidence**: 0.2593

## Generated Files

- Confusion Matrix: `ssvep_SSVEPFormer_confusion_matrix.png`
- Evaluation Metrics: `ssvep_SSVEPFormer_metrics.json`
- This Report: `ssvep_SSVEPFormer_evaluation_report.md`
