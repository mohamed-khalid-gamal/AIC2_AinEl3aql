# MI EnhancedFeatureClassifier Model Evaluation Report

## Model Information

- **Task**: MI
- **Model Type**: EnhancedFeatureClassifier
- **Evaluation Date**: 2025-06-30 23:43:34

## Validation Set Results

- **Accuracy**: 0.7000
- **F1 Score (Weighted)**: 0.6967
- **F1 Score (Macro)**: 0.6989
- **Precision (Weighted)**: 0.7413
- **Recall (Weighted)**: 0.7000

### Per-Class Metrics

- **Class Left**:
  - Precision: 0.8421
  - Recall: 0.5714
  - F1: 0.6809
- **Class Right**:
  - Precision: 0.6129
  - Recall: 0.8636
  - F1: 0.7170

## Training History

- **Final Training Accuracy**: 0.6104
- **Best Validation Accuracy**: 0.7000
- **Final Validation F1**: 0.6992
- **Training Epochs**: 48

## Detailed Classification Report

```
              precision    recall  f1-score   support

        Left       0.84      0.57      0.68        28
       Right       0.61      0.86      0.72        22

    accuracy                           0.70        50
   macro avg       0.73      0.72      0.70        50
weighted avg       0.74      0.70      0.70        50

```

## Prediction Confidence Statistics

- **Mean Confidence**: 0.5727
- **Std Confidence**: 0.0543
- **Min Confidence**: 0.5056
- **Max Confidence**: 0.7107

## Generated Files

- Confusion Matrix: `mi_EnhancedFeatureClassifier_confusion_matrix.png`
- Evaluation Metrics: `mi_EnhancedFeatureClassifier_metrics.json`
- This Report: `mi_EnhancedFeatureClassifier_evaluation_report.md`
