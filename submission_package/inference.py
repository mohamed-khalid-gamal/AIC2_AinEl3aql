#!/usr/bin/env python3
"""
EEG Model Inference Script
Loads trained models and generates predictions for test data.
"""

import argparse
import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
from train import (EEGNet, DeepConvNet, EnhancedFeatureClassifier, 
                   SSVEPFormer, BiLSTMClassifier, create_model)


def load_model_and_config(model_path, config_path):
    """Load trained model and its configuration."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_type = config['model_type']
    # Extract from nested config structure
    model_config_data = config['config']
    input_shape = model_config_data['input_shape']
    num_classes = model_config_data['num_classes']
    
    # Handle input_shape format - convert list to appropriate format
    if isinstance(input_shape, list):
        if len(input_shape) == 1:
            # For feature-based models, use the single dimension
            input_shape = input_shape[0]
        else:
            # For multi-dimensional data (like EEG channels x samples)
            input_shape = tuple(input_shape)
    
    print(f"Processed input_shape: {input_shape}, type: {type(input_shape)}")
    
    # Create model architecture - filter out training-specific parameters
    model_config = {k: v for k, v in model_config_data.items() 
                   if k not in ['epochs', 'batch_size', 'lr', 'patience', 'device', 'input_shape', 'num_classes', 'n_features']}
    
    model = create_model(model_type, input_shape, num_classes, **model_config)
    
    # Load model weights/state
    if model_path.endswith('.pth'):  # PyTorch model
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:  # Sklearn model
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        model = checkpoint['model']
    
    return model, config


def predict_sklearn_model(model, X_test):
    """Generate predictions using sklearn model."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    confidence = probabilities.max(axis=1)
    
    return predictions, confidence


def predict_pytorch_model(model, X_test, batch_size=64, device='cuda'):
    """Generate predictions using PyTorch model."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Create data loader
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for batch_x, in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probabilities = F.softmax(outputs, dim=1)
            
            predictions = torch.argmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_confidences)


def generate_task_predictions(task_data, model_path, config_path, label_encoder, 
                             batch_size=64, device='cuda'):
    """Generate predictions for a specific task."""
    
    print(f"Loading model from {model_path}")
    model, config = load_model_and_config(model_path, config_path)
    
    X_test = task_data['X_test']
    test_ids = task_data['ids_test']
    model_type = config['model_type']
    
    print(f"Generating predictions using {model_type}")
    print(f"Test samples: {len(X_test)}")
    
    # Generate predictions
    if model_type in ['LDA', 'SVM', 'RandomForest']:
        predictions, confidence = predict_sklearn_model(model, X_test)
    else:
        predictions, confidence = predict_pytorch_model(model, X_test, batch_size, device)
    
    # Convert predictions back to original labels
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions)
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'id': test_ids,
        'label': predictions,
        'confidence': confidence
    })
    
    return pred_df


def main():
    parser = argparse.ArgumentParser(description='EEG Model Inference')
    
    # Required paths
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained models directory')
    parser.add_argument('--output_path', type=str, default='./submission.csv',
                       help='Path to save prediction CSV')
    
    # Task selection
    parser.add_argument('--task', type=str, choices=['MI', 'SSVEP', 'both'], default='both',
                       help='Which task to predict')
    
    # Model specification (if not using default naming)
    parser.add_argument('--mi_model_file', type=str, default=None,
                       help='Specific MI model file (optional)')
    parser.add_argument('--ssvep_model_file', type=str, default=None,
                       help='Specific SSVEP model file (optional)')
    
    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Load label encoders
    print("Loading label encoders...")
    with open(os.path.join(args.data_path, 'label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
    
    all_predictions = []
    
    # Process MI task
    if args.task in ['MI', 'both']:
        try:
            print(f"\n{'='*50}")
            print("Processing MI Task")
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
                # Auto-detect MI model files
                mi_files = [f for f in os.listdir(args.model_path) if f.startswith('mi_') and f.endswith(('.pth', '.pkl'))]
                if not mi_files:
                    raise FileNotFoundError("No MI model files found")
                
                mi_model_file = mi_files[0]  # Take the first one found
                mi_model_path = os.path.join(args.model_path, mi_model_file)
                # Extract base name and construct config path
                base_name = mi_model_file.replace('_model.pth', '').replace('_model.pkl', '').replace('.pth', '').replace('.pkl', '')
                mi_config_path = os.path.join(args.model_path, f"{base_name}_config.json")
            
            # Generate MI predictions
            mi_predictions = generate_task_predictions(
                mi_data, mi_model_path, mi_config_path, 
                label_encoders['mi'], args.batch_size, args.device
            )
            mi_predictions['task'] = 'MI'
            all_predictions.append(mi_predictions)
            
            print(f"✓ MI predictions generated: {len(mi_predictions)} samples")
            print(f"  Confidence stats: mean={mi_predictions['confidence'].mean():.3f}, "
                  f"std={mi_predictions['confidence'].std():.3f}")
            
        except (FileNotFoundError, KeyError) as e:
            print(f"⚠ MI prediction failed: {e}")
    
    # Process SSVEP task
    if args.task in ['SSVEP', 'both']:
        try:
            print(f"\n{'='*50}")
            print("Processing SSVEP Task")
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
                # Auto-detect SSVEP model files - prefer EnhancedFeatureClassifier if available
                ssvep_files = [f for f in os.listdir(args.model_path) if f.startswith('ssvep_') and f.endswith(('.pth', '.pkl'))]
                if not ssvep_files:
                    raise FileNotFoundError("No SSVEP model files found")
                
                # Prefer EnhancedFeatureClassifier if available
                enhanced_files = [f for f in ssvep_files if 'EnhancedFeatureClassifier' in f]
                if enhanced_files:
                    ssvep_model_file = enhanced_files[0]
                    print(f"✓ Using EnhancedFeatureClassifier for SSVEP: {ssvep_model_file}")
                else:
                    ssvep_model_file = ssvep_files[0]  # Take the first one found
                    print(f"✓ Using SSVEP model: {ssvep_model_file}")
                
                ssvep_model_path = os.path.join(args.model_path, ssvep_model_file)
                # Extract base name and construct config path
                base_name = ssvep_model_file.replace('_model.pth', '').replace('_model.pkl', '').replace('.pth', '').replace('.pkl', '')
                ssvep_config_path = os.path.join(args.model_path, f"{base_name}_config.json")
            
            # Generate SSVEP predictions
            ssvep_predictions = generate_task_predictions(
                ssvep_data, ssvep_model_path, ssvep_config_path,
                label_encoders['ssvep'], args.batch_size, args.device
            )
            ssvep_predictions['task'] = 'SSVEP'
            all_predictions.append(ssvep_predictions)
            
            print(f"✓ SSVEP predictions generated: {len(ssvep_predictions)} samples")
            print(f"  Confidence stats: mean={ssvep_predictions['confidence'].mean():.3f}, "
                  f"std={ssvep_predictions['confidence'].std():.3f}")
            
        except (FileNotFoundError, KeyError) as e:
            print(f"⚠ SSVEP prediction failed: {e}")
    
    # Combine and save predictions
    if all_predictions:
        print(f"\n{'='*50}")
        print("Combining and Saving Predictions")
        print(f"{'='*50}")
        
        # Combine all predictions
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        final_predictions = final_predictions.sort_values('id').reset_index(drop=True)
        
        # Create submission format (only id and label columns)
        submission = final_predictions[['id', 'label']].copy()
        
        # Save submission file
        submission.to_csv(args.output_path, index=False)
        
        print(f"✓ Submission saved to: {args.output_path}")
        print(f"  Total predictions: {len(submission)}")
        print(f"  Tasks included: {final_predictions['task'].unique()}")
        
        # Save detailed predictions (with confidence and task info)
        detailed_path = args.output_path.replace('.csv', '_detailed.csv')
        final_predictions.to_csv(detailed_path, index=False)
        print(f"✓ Detailed predictions saved to: {detailed_path}")
        
        # Display prediction statistics
        print("\nPrediction Summary:")
        for task in final_predictions['task'].unique():
            task_preds = final_predictions[final_predictions['task'] == task]
            print(f"  {task}: {len(task_preds)} predictions")
            print(f"    Label distribution: {task_preds['label'].value_counts().to_dict()}")
            print(f"    Avg confidence: {task_preds['confidence'].mean():.3f}")
        
        print("\nSubmission Preview:")
        print(submission.head(10))
        print("...")
        print(submission.tail(10))
        
    else:
        print("⚠ No predictions were generated!")


if __name__ == "__main__":
    main()
