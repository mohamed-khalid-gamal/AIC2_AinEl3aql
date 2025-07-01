# SSVEP EnhancedFeatureClassifier Model Evaluation Report

## Model Information

- **Task**: SSVEP
- **Model Type**: EnhancedFeatureClassifier
- **Evaluation Date**: 2025-06-30 23:43:35

## Validation Set Results

- **Accuracy**: 0.4000
- **F1 Score (Weighted)**: 0.4094
- **F1 Score (Macro)**: 0.3983
- **Precision (Weighted)**: 0.4599
- **Recall (Weighted)**: 0.4000

### Per-Class Metrics

- **Class Backward**:
  - Precision: 0.7778
  - Recall: 0.5000
  - F1: 0.6087
- **Class Forward**:
  - Precision: 0.3571
  - Recall: 0.4167
  - F1: 0.3846
- **Class Left**:
  - Precision: 0.2727
  - Recall: 0.4286
  - F1: 0.3333
- **Class Right**:
  - Precision: 0.4000
  - Recall: 0.2000
  - F1: 0.2667

## Training History

- **Final Training Accuracy**: 0.5758
- **Best Validation Accuracy**: 0.4000
- **Final Validation F1**: 0.3868
- **Training Epochs**: 16

## Detailed Classification Report

```
              precision    recall  f1-score   support

    Backward       0.78      0.50      0.61        14
     Forward       0.36      0.42      0.38        12
        Left       0.27      0.43      0.33        14
       Right       0.40      0.20      0.27        10

    accuracy                           0.40        50
   macro avg       0.45      0.39      0.40        50
weighted avg       0.46      0.40      0.41        50

```

## Prediction Confidence Statistics

- **Mean Confidence**: 0.4577
- **Std Confidence**: 0.1984
- **Min Confidence**: 0.2773
- **Max Confidence**: 0.9918

## Generated Files

- Confusion Matrix: `ssvep_EnhancedFeatureClassifier_confusion_matrix.png`
- Evaluation Metrics: `ssvep_EnhancedFeatureClassifier_metrics.json`
- This Report: `ssvep_EnhancedFeatureClassifier_evaluation_report.md`
