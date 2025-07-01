#!/usr/bin/env python3
"""
EEG Model Evaluation Script
Evaluates trained models on validation data and generates comprehensive reports.
"""

import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Import inference functions
from inference import load_model_and_config, predict_sklearn_model, predict_pytorch_model


def calculate_metrics(y_true, y_pred, y_prob=None, labels=None):
    """Calculate comprehensive evaluation metrics."""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Add per-class metrics
    if labels is not None:
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, label in enumerate(labels):
            if i < len(precision_per_class):
                metrics[f'precision_class_{label}'] = precision_per_class[i]
                metrics[f'recall_class_{label}'] = recall_per_class[i]
                metrics[f'f1_class_{label}'] = f1_per_class[i]
    
    # Add AUC if probabilities are available and it's binary/multiclass
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            else:  # Multiclass
                metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except (ValueError, IndexError):
            pass  # Skip AUC if not applicable
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot and save confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_roc_curves(y_true, y_prob, labels, title, save_path):
    """Plot ROC curves for multiclass classification."""
    
    n_classes = len(labels)
    
    if n_classes == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {title}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    elif n_classes > 2:  # Multiclass
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import auc
        
        # Binarize labels
        y_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            if i < y_prob.shape[1]:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {labels[i]} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {title}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_model_on_data(model, config, X, y, label_encoder=None, 
                          batch_size=64, device='cuda'):
    """Evaluate model on given data."""
    
    model_type = config['model_type']
    
    # Convert labels to proper integer format - same fix as in training
    print(f"Original label types in evaluation: y={type(y[0]) if len(y) > 0 else 'empty'}")
    y = np.array([int(x) for x in y], dtype=np.int64)
    print(f"Converted label types in evaluation: y={y.dtype}")
    
    # Generate predictions
    if model_type in ['LDA', 'SVM', 'RandomForest']:
        y_pred, confidence = predict_sklearn_model(model, X)
        y_prob = model.predict_proba(X)
    else:
        y_pred, confidence = predict_pytorch_model(model, X, batch_size, device)
        # For PyTorch models, we need to get probabilities separately
        y_prob = None  # Would need to modify predict_pytorch_model to return probabilities
    
    # Convert predictions back to original labels if needed
    if label_encoder is not None:
        y_pred_original = label_encoder.inverse_transform(y_pred)
        y_true_original = label_encoder.inverse_transform(y)
        labels = label_encoder.classes_
    else:
        y_pred_original = y_pred
        y_true_original = y
        labels = np.unique(y)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_original, y_pred_original, y_prob, labels)
    
    return {
        'metrics': metrics,
        'y_true': y_true_original,
        'y_pred': y_pred_original,
        'y_prob': y_prob,
        'confidence': confidence,
        'labels': labels
    }


def perform_cross_validation(task_data, model_type, model_config, cv_folds=5):
    """Perform cross-validation evaluation."""
    
    from train import create_model
    from sklearn.model_selection import cross_validate
    
    X = task_data['X_train']
    y = task_data['y_train']
    
    # Convert labels to proper integer format - same fix as in training
    print(f"Original label types in CV: y={type(y[0]) if len(y) > 0 else 'empty'}")
    y = np.array([int(x) for x in y], dtype=np.int64)
    print(f"Converted label types in CV: y={y.dtype}")
    
    print(f"Performing {cv_folds}-fold cross-validation...")
    
    if model_type in ['LDA', 'SVM', 'RandomForest']:
        # Create sklearn model
        model = create_model(model_type, X.shape[1], len(np.unique(y)), **model_config)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
            return_train_score=True
        )
        
        cv_metrics = {
            'test_accuracy': cv_results['test_accuracy'],
            'test_f1': cv_results['test_f1_weighted'],
            'test_precision': cv_results['test_precision_weighted'],
            'test_recall': cv_results['test_recall_weighted'],
            'train_accuracy': cv_results['train_accuracy'],
            'train_f1': cv_results['train_f1_weighted'],
        }
        
        # Calculate statistics
        cv_stats = {}
        for metric, scores in cv_metrics.items():
            cv_stats[f'{metric}_mean'] = np.mean(scores)
            cv_stats[f'{metric}_std'] = np.std(scores)
        
        return cv_stats
    
    else:
        print("Cross-validation not implemented for PyTorch models in this version.")
        return {}


def generate_classification_report_text(y_true, y_pred, labels):
    """Generate detailed classification report text."""
    
    report = classification_report(y_true, y_pred, target_names=[str(l) for l in labels])
    return report


def create_evaluation_report(task, model_type, evaluation_results, cv_results, 
                           metrics_history, save_path):
    """Create comprehensive evaluation report."""
    
    report_path = os.path.join(save_path, f'{task.lower()}_{model_type}_evaluation_report.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# {task} {model_type} Model Evaluation Report\n\n")
        
        # Model Information
        f.write("## Model Information\n\n")
        f.write(f"- **Task**: {task}\n")
        f.write(f"- **Model Type**: {model_type}\n")
        f.write(f"- **Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Validation Results
        f.write("## Validation Set Results\n\n")
        metrics = evaluation_results['metrics']
        f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
        f.write(f"- **F1 Score (Weighted)**: {metrics['f1_weighted']:.4f}\n")
        f.write(f"- **F1 Score (Macro)**: {metrics['f1_macro']:.4f}\n")
        f.write(f"- **Precision (Weighted)**: {metrics['precision_weighted']:.4f}\n")
        f.write(f"- **Recall (Weighted)**: {metrics['recall_weighted']:.4f}\n")
        
        if 'auc' in metrics:
            f.write(f"- **AUC**: {metrics['auc']:.4f}\n")
        if 'auc_ovr' in metrics:
            f.write(f"- **AUC (OvR)**: {metrics['auc_ovr']:.4f}\n")
        
        f.write("\n")
        
        # Per-class metrics
        f.write("### Per-Class Metrics\n\n")
        labels = evaluation_results['labels']
        for label in labels:
            if f'f1_class_{label}' in metrics:
                f.write(f"- **Class {label}**:\n")
                f.write(f"  - Precision: {metrics[f'precision_class_{label}']:.4f}\n")
                f.write(f"  - Recall: {metrics[f'recall_class_{label}']:.4f}\n")
                f.write(f"  - F1: {metrics[f'f1_class_{label}']:.4f}\n")
        f.write("\n")
        
        # Cross-validation results
        if cv_results:
            f.write("## Cross-Validation Results\n\n")
            f.write(f"- **Test Accuracy**: {cv_results.get('test_accuracy_mean', 0):.4f} ± {cv_results.get('test_accuracy_std', 0):.4f}\n")
            f.write(f"- **Test F1**: {cv_results.get('test_f1_mean', 0):.4f} ± {cv_results.get('test_f1_std', 0):.4f}\n")
            f.write(f"- **Test Precision**: {cv_results.get('test_precision_mean', 0):.4f} ± {cv_results.get('test_precision_std', 0):.4f}\n")
            f.write(f"- **Test Recall**: {cv_results.get('test_recall_mean', 0):.4f} ± {cv_results.get('test_recall_std', 0):.4f}\n\n")
        
        # Training history (if available)
        if metrics_history and 'train_history' in metrics_history:
            history = metrics_history['train_history']
            if 'val_acc' in history and len(history['val_acc']) > 0:
                f.write("## Training History\n\n")
                f.write(f"- **Final Training Accuracy**: {history['train_acc'][-1]:.4f}\n")
                f.write(f"- **Best Validation Accuracy**: {max(history['val_acc']):.4f}\n")
                f.write(f"- **Final Validation F1**: {history['val_f1'][-1]:.4f}\n")
                f.write(f"- **Training Epochs**: {len(history['train_acc'])}\n\n")
        
        # Classification report
        f.write("## Detailed Classification Report\n\n")
        f.write("```\n")
        class_report = generate_classification_report_text(
            evaluation_results['y_true'], 
            evaluation_results['y_pred'], 
            evaluation_results['labels']
        )
        f.write(class_report)
        f.write("\n```\n\n")
        
        # Confidence statistics
        f.write("## Prediction Confidence Statistics\n\n")
        confidence = evaluation_results['confidence']
        f.write(f"- **Mean Confidence**: {np.mean(confidence):.4f}\n")
        f.write(f"- **Std Confidence**: {np.std(confidence):.4f}\n")
        f.write(f"- **Min Confidence**: {np.min(confidence):.4f}\n")
        f.write(f"- **Max Confidence**: {np.max(confidence):.4f}\n\n")
        
        # Files generated
        f.write("## Generated Files\n\n")
        f.write(f"- Confusion Matrix: `{task.lower()}_{model_type}_confusion_matrix.png`\n")
        if evaluation_results['y_prob'] is not None:
            f.write(f"- ROC Curves: `{task.lower()}_{model_type}_roc_curves.png`\n")
        f.write(f"- Evaluation Metrics: `{task.lower()}_{model_type}_metrics.json`\n")
        f.write(f"- This Report: `{task.lower()}_{model_type}_evaluation_report.md`\n")
    
    print(f"✓ Evaluation report saved: {report_path}")
    return report_path


def evaluate_task_model(task, task_data, model_path, config_path, label_encoder,
                       model_config, save_path, batch_size=64, device='cuda', cv_folds=5):
    """Evaluate model for a specific task."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating {task} Model")
    print(f"{'='*50}")
    
    # Load model and configuration
    model, config = load_model_and_config(model_path, config_path)
    model_type = config['model_type']
    
    print(f"Model: {model_type}")
    print(f"Validation samples: {len(task_data['X_val'])}")
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_results = evaluate_model_on_data(
        model, config, task_data['X_val'], task_data['y_val'],
        label_encoder, batch_size, device
    )
    
    print(f"✓ Validation Accuracy: {val_results['metrics']['accuracy']:.4f}")
    print(f"✓ Validation F1 (Weighted): {val_results['metrics']['f1_weighted']:.4f}")
    
    # Perform cross-validation (for sklearn models)
    cv_results = {}
    if model_type in ['LDA', 'SVM', 'RandomForest']:
        try:
            cv_results = perform_cross_validation(task_data, model_type, model_config, cv_folds)
            if cv_results:
                print(f"✓ CV Accuracy: {cv_results['test_accuracy_mean']:.4f} ± {cv_results['test_accuracy_std']:.4f}")
        except Exception as e:
            print(f"⚠ Cross-validation failed: {e}")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Confusion matrix
    cm_path = os.path.join(save_path, f'{task.lower()}_{model_type}_confusion_matrix.png')
    cm = plot_confusion_matrix(
        val_results['y_true'], val_results['y_pred'], val_results['labels'],
        f'{task} {model_type}', cm_path
    )
    
    # ROC curves (if probabilities available)
    if val_results['y_prob'] is not None:
        roc_path = os.path.join(save_path, f'{task.lower()}_{model_type}_roc_curves.png')
        plot_roc_curves(
            val_results['y_true'], val_results['y_prob'], val_results['labels'],
            f'{task} {model_type}', roc_path
        )
    
    # Save metrics
    metrics_path = os.path.join(save_path, f'{task.lower()}_{model_type}_metrics.json')
    all_metrics = {
        'validation_metrics': val_results['metrics'],
        'cross_validation_metrics': cv_results,
        'confusion_matrix': cm.tolist(),
        'model_info': config
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Generate comprehensive report
    metrics_history = config.get('metrics', {})
    report_path = create_evaluation_report(
        task, model_type, val_results, cv_results, metrics_history, save_path
    )
    
    return {
        'validation_results': val_results,
        'cv_results': cv_results,
        'report_path': report_path,
        'metrics_path': metrics_path
    }


def main():
    parser = argparse.ArgumentParser(description='EEG Model Evaluation')
    
    # Required paths
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained models directory')
    parser.add_argument('--output_path', type=str, default='./results',
                       help='Path to save evaluation results')
    
    # Task selection
    parser.add_argument('--task', type=str, choices=['MI', 'SSVEP', 'both'], default='both',
                       help='Which task to evaluate')
    
    # Model specification
    parser.add_argument('--mi_model_file', type=str, default=None,
                       help='Specific MI model file (optional)')
    parser.add_argument('--ssvep_model_file', type=str, default=None,
                       help='Specific SSVEP model file (optional)')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for evaluation')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load label encoders
    print("Loading label encoders...")
    with open(os.path.join(args.data_path, 'label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
    
    evaluation_results = {}
    
    # Evaluate MI task
    if args.task in ['MI', 'both']:
        try:
            print(f"\n{'='*50}")
            print("Evaluating MI Task")
            print(f"{'='*50}")
            
            # Load MI data
            with open(os.path.join(args.data_path, 'mi_data.pkl'), 'rb') as f:
                mi_data = pickle.load(f)
            
            # Find MI model files
            if args.mi_model_file:
                mi_model_path = os.path.join(args.model_path, args.mi_model_file)
                # Extract base name and construct config path
                base_name = args.mi_model_file.replace('_model.pth', '').replace('_model.pkl', '').replace('.pth', '').replace('.pkl', '')
                mi_config_path = os.path.join(args.model_path, f"{base_name}_config.json")
            else:
                mi_files = [f for f in os.listdir(args.model_path) if f.startswith('mi_') and f.endswith(('.pth', '.pkl'))]
                if not mi_files:
                    raise FileNotFoundError("No MI model files found")
                
                mi_model_file = mi_files[0]
                mi_model_path = os.path.join(args.model_path, mi_model_file)
                # Extract base name and construct config path
                base_name = mi_model_file.replace('_model.pth', '').replace('_model.pkl', '').replace('.pth', '').replace('.pkl', '')
                mi_config_path = os.path.join(args.model_path, f"{base_name}_config.json")
            
            # Evaluate MI model
            mi_eval_results = evaluate_task_model(
                'MI', mi_data, mi_model_path, mi_config_path,
                label_encoders['mi'], {}, args.output_path,
                args.batch_size, args.device, args.cv_folds
            )
            
            evaluation_results['MI'] = mi_eval_results
            print(f"✓ MI evaluation completed")
            
        except (FileNotFoundError, KeyError) as e:
            print(f"⚠ MI evaluation failed: {e}")
    
    # Evaluate SSVEP task
    if args.task in ['SSVEP', 'both']:
        try:
            print(f"\n{'='*50}")
            print("Evaluating SSVEP Task")
            print(f"{'='*50}")
            
            # Load SSVEP data
            with open(os.path.join(args.data_path, 'ssvep_data.pkl'), 'rb') as f:
                ssvep_data = pickle.load(f)
            
            # Find SSVEP model files
            if args.ssvep_model_file:
                ssvep_model_path = os.path.join(args.model_path, args.ssvep_model_file)
                # Extract base name and construct config path
                base_name = args.ssvep_model_file.replace('_model.pth', '').replace('_model.pkl', '').replace('.pth', '').replace('.pkl', '')
                ssvep_config_path = os.path.join(args.model_path, f"{base_name}_config.json")
            else:
                ssvep_files = [f for f in os.listdir(args.model_path) if f.startswith('ssvep_') and f.endswith(('.pth', '.pkl'))]
                if not ssvep_files:
                    raise FileNotFoundError("No SSVEP model files found")
                
                ssvep_model_file = ssvep_files[0]
                ssvep_model_path = os.path.join(args.model_path, ssvep_model_file)
                # Extract base name and construct config path
                base_name = ssvep_model_file.replace('_model.pth', '').replace('_model.pkl', '').replace('.pth', '').replace('.pkl', '')
                ssvep_config_path = os.path.join(args.model_path, f"{base_name}_config.json")
            
            # Evaluate SSVEP model
            ssvep_eval_results = evaluate_task_model(
                'SSVEP', ssvep_data, ssvep_model_path, ssvep_config_path,
                label_encoders['ssvep'], {}, args.output_path,
                args.batch_size, args.device, args.cv_folds
            )
            
            evaluation_results['SSVEP'] = ssvep_eval_results
            print(f"✓ SSVEP evaluation completed")
            
        except (FileNotFoundError, KeyError) as e:
            print(f"⚠ SSVEP evaluation failed: {e}")
    
    # Generate summary report
    if evaluation_results:
        print(f"\n{'='*50}")
        print("Generating Summary Report")
        print(f"{'='*50}")
        
        summary_path = os.path.join(args.output_path, 'evaluation_summary.md')
        
        with open(summary_path, 'w') as f:
            f.write("# EEG Model Evaluation Summary\n\n")
            f.write(f"**Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for task, results in evaluation_results.items():
                val_metrics = results['validation_results']['metrics']
                f.write(f"## {task} Task Results\n\n")
                f.write(f"- **Accuracy**: {val_metrics['accuracy']:.4f}\n")
                f.write(f"- **F1 Score (Weighted)**: {val_metrics['f1_weighted']:.4f}\n")
                f.write(f"- **F1 Score (Macro)**: {val_metrics['f1_macro']:.4f}\n")
                
                if results['cv_results']:
                    cv = results['cv_results']
                    f.write(f"- **CV Accuracy**: {cv.get('test_accuracy_mean', 0):.4f} ± {cv.get('test_accuracy_std', 0):.4f}\n")
                
                f.write(f"- **Report**: [{task} Evaluation Report]({os.path.basename(results['report_path'])})\n\n")
        
        print(f"✓ Summary report saved: {summary_path}")
        print(f"✓ All evaluation results saved to: {args.output_path}")
    
    else:
        print("⚠ No evaluation results to summarize!")


if __name__ == "__main__":
    main()
