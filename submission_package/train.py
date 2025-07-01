#!/usr/bin/env python3
"""
EEG Model Training Script
Supports multiple model architectures for both MI and SSVEP tasks.
"""

import argparse
import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# ================================
# PYTORCH MODELS
# ================================

class EEGNet(nn.Module):
    """EEGNet architecture for EEG classification."""
    
    def __init__(self, channels, samples, num_classes, dropout=0.25):
        super(EEGNet, self).__init__()
        
        # Layer 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(16)
        
        # Layer 2: Spatial Convolution (Depthwise)
        self.depthwise = nn.Conv2d(16, 32, (channels, 1), groups=16)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 3: Separable Convolution
        self.separable = nn.Conv2d(32, 32, (1, 16), padding=(0, 8))
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Calculate output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, channels, samples)
            dummy_output = self._forward_features(dummy_input)
            self.feature_size = dummy_output.view(1, -1).shape[1]
            
        self.classifier = nn.Linear(self.feature_size, num_classes)
        
    def _forward_features(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.elu(self.bn3(self.separable(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        return x
        
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DeepConvNet(nn.Module):
    """DeepConvNet architecture for EEG classification."""
    
    def __init__(self, channels, samples, num_classes, dropout=0.5):
        super(DeepConvNet, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (channels, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            
            # Block 2
            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            
            # Block 3
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            
            # Block 4
            nn.Conv2d(100, 200, (1, 5)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        
        # Calculate output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, channels, samples)
            dummy_output = self.conv_blocks(dummy_input)
            self.feature_size = dummy_output.view(1, -1).shape[1]
            
        self.classifier = nn.Linear(self.feature_size, num_classes)
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class AttentionModule(nn.Module):
    """Attention mechanism for feature enhancement."""
    
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights


class EnhancedFeatureClassifier(nn.Module):
    """MLP-based classifier with attention for feature data."""
    
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(EnhancedFeatureClassifier, self).__init__()

        hidden_dim1 = max(256, input_dim // 4)
        hidden_dim2 = max(128, input_dim // 8)
        hidden_dim3 = max(64, input_dim // 16)

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),

            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6)
        )

        self.attention = AttentionModule(hidden_dim3)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim3, hidden_dim3 // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim3 // 2, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.attention(x)
        return self.classifier(x)


class SSVEPFormer(nn.Module):
    """Transformer-based SSVEP classifier."""
    
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(SSVEPFormer, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier."""
    
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=1, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, seq_len=1, input_dim)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)  # Concatenate forward and backward
        return self.fc(h)


# ================================
# MODEL FACTORY
# ================================

def create_model(model_type, input_shape, num_classes, **kwargs):
    """Create model based on type and parameters."""
    
    if model_type == 'LDA':
        # Filter kwargs for LDA
        lda_kwargs = {k: v for k, v in kwargs.items() if k in ['solver', 'shrinkage', 'priors', 'n_components', 'store_covariance', 'tol']}
        return LinearDiscriminantAnalysis(**lda_kwargs)
    elif model_type == 'SVM':
        # Filter kwargs for SVM
        svm_kwargs = {k: v for k, v in kwargs.items() if k in ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking', 'tol', 'cache_size', 'class_weight', 'verbose', 'max_iter', 'decision_function_shape', 'break_ties', 'random_state']}
        return SVC(probability=True, **svm_kwargs)
    elif model_type == 'RandomForest':
        # Filter kwargs for RandomForest
        rf_kwargs = {k: v for k, v in kwargs.items() if k in ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples']}
        return RandomForestClassifier(**rf_kwargs)
    elif model_type == 'EEGNet':
        channels, samples = input_shape
        # Filter kwargs for EEGNet
        eegnet_kwargs = {k: v for k, v in kwargs.items() if k in ['dropout']}
        return EEGNet(channels, samples, num_classes, **eegnet_kwargs)
    elif model_type == 'DeepConvNet':
        channels, samples = input_shape
        # Filter kwargs for DeepConvNet
        deepconv_kwargs = {k: v for k, v in kwargs.items() if k in ['dropout']}
        return DeepConvNet(channels, samples, num_classes, **deepconv_kwargs)
    elif model_type == 'EnhancedFeatureClassifier':
        input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        # Filter kwargs for EnhancedFeatureClassifier
        efc_kwargs = {k: v for k, v in kwargs.items() if k in ['dropout']}
        return EnhancedFeatureClassifier(input_dim, num_classes, **efc_kwargs)
    elif model_type == 'SSVEPFormer':
        input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        # Filter kwargs for SSVEPFormer
        transformer_kwargs = {k: v for k, v in kwargs.items() if k in ['d_model', 'nhead', 'num_layers', 'dropout']}
        return SSVEPFormer(input_dim, num_classes, **transformer_kwargs)
    elif model_type == 'BiLSTMClassifier':
        input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        # Filter kwargs for BiLSTMClassifier
        lstm_kwargs = {k: v for k, v in kwargs.items() if k in ['hidden_dim', 'num_layers', 'dropout']}
        return BiLSTMClassifier(input_dim, num_classes, **lstm_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ================================
# TRAINING UTILITIES
# ================================

def train_sklearn_model(model, X_train, y_train, X_val, y_val):
    """Train sklearn model."""
    print("Training sklearn model...")
    
    # Convert labels to proper integer format
    print(f"Original label types: y_train={type(y_train[0]) if len(y_train) > 0 else 'empty'}")
    if y_val is not None:
        print(f"y_val={type(y_val[0]) if len(y_val) > 0 else 'empty'}")
    
    # Force conversion from object type to int64
    y_train = np.asarray(y_train, dtype=np.int64)
    if y_val is not None:
        y_val = np.asarray(y_val, dtype=np.int64)
    
    print(f"Converted label types: y_train={y_train.dtype}")
    if y_val is not None:
        print(f"y_val={y_val.dtype}")
    
    model.fit(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        return {'val_acc': val_acc, 'val_f1': val_f1}
    
    return {}


def train_pytorch_model(model, X_train, y_train, X_val, y_val, 
                       epochs=50, batch_size=32, lr=1e-3, patience=15, device='cuda'):
    """Train PyTorch model with early stopping."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training PyTorch model on device: {device}")
    
    # Convert labels to proper integer format
    print(f"Original label types: y_train={type(y_train[0]) if len(y_train) > 0 else 'empty'}")
    if y_val is not None:
        print(f"y_val={type(y_val[0]) if len(y_val) > 0 else 'empty'}")
    
    # Force conversion from object type to int64 - use manual conversion for robustness
    y_train = np.array([int(x) for x in y_train], dtype=np.int64)
    if y_val is not None:
        y_val = np.array([int(x) for x in y_val], dtype=np.int64)
    
    print(f"Converted label types: y_train={y_train.dtype}")
    if y_val is not None:
        print(f"y_val={y_val.dtype}")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_acc = 0
    patience_counter = 0
    train_history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # L2 regularization
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 0.0001 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)
        
        train_acc = correct_train / total_train
        train_loss = running_loss / len(train_loader)
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        
        # Validation phase
        val_acc = 0
        val_f1 = 0
        if val_loader:
            model.eval()
            correct_val = 0
            total_val = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs, 1)
                    
                    correct_val += (predicted == batch_y).sum().item()
                    total_val += batch_y.size(0)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.numpy())
            
            val_acc = correct_val / total_val
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            train_history['val_acc'].append(val_acc)
            train_history['val_f1'].append(val_f1)
        
        scheduler.step()
        
        # Logging
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if val_loader:
                print(f"  Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loader and val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_temp.pth')
        elif val_loader:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if val_loader and os.path.exists('best_model_temp.pth'):
        model.load_state_dict(torch.load('best_model_temp.pth'))
        os.remove('best_model_temp.pth')
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'best_val_acc': best_val_acc,
        'train_history': train_history
    }


def save_model_and_config(model, model_type, task, config, metrics, save_path):
    """Save trained model and configuration."""
    
    model_info = {
        'model_type': model_type,
        'task': task,
        'config': config,
        'metrics': metrics,
        'input_shape': config.get('input_shape'),
        'num_classes': config.get('num_classes')
    }
    
    # Save model
    if hasattr(model, 'state_dict'):  # PyTorch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_info': model_info
        }, os.path.join(save_path, f'{task.lower()}_{model_type}_model.pth'))
    else:  # Sklearn model
        import pickle
        with open(os.path.join(save_path, f'{task.lower()}_{model_type}_model.pkl'), 'wb') as f:
            pickle.dump({
                'model': model,
                'model_info': model_info
            }, f)
    
    # Save config separately
    with open(os.path.join(save_path, f'{task.lower()}_{model_type}_config.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✓ Saved {task} {model_type} model and config")


def train_task_model(task_data, task, model_type, model_config, train_config, save_path):
    """Train model for a specific task."""
    
    print(f"\n{'='*50}")
    print(f"Training {model_type} for {task} Task")
    print(f"{'='*50}")
    
    X_train = task_data['X_train']
    y_train = task_data['y_train']
    X_val = task_data['X_val'] 
    y_val = task_data['y_val']
    num_classes = task_data['n_classes']
    n_features = task_data['n_features']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {n_features}")
    print(f"Classes: {num_classes}")
    
    # Determine input shape and validate model compatibility
    feature_type = task_data.get('feature_type', 'unknown')
    
    if model_type in ['EEGNet', 'DeepConvNet']:
        # These models require raw EEG data, not extracted features
        if feature_type != 'raw':
            print(f"⚠️  Warning: {model_type} expects raw EEG data, but found {feature_type} features.")
            print(f"   Switching to EnhancedFeatureClassifier which works with extracted features.")
            model_type = 'EnhancedFeatureClassifier'
            input_shape = (n_features,)
        else:
            # For raw EEG data - need to reconstruct shape
            # Assuming 8 channels and calculate samples from features
            estimated_samples = n_features // 8
            input_shape = (8, estimated_samples)
    else:
        input_shape = (n_features,)
    
    # Create model
    model = create_model(model_type, input_shape, num_classes, **model_config)
    
    print(f"Model created with input shape: {input_shape}")
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    # Train model
    if model_type in ['LDA', 'SVM', 'RandomForest']:
        metrics = train_sklearn_model(model, X_train, y_train, X_val, y_val)
    else:
        metrics = train_pytorch_model(model, X_train, y_train, X_val, y_val, **train_config)
    
    # Save model
    config = {
        **model_config,
        **train_config,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'n_features': n_features
    }
    
    save_model_and_config(model, model_type, task, config, metrics, save_path)
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='EEG Model Training')
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data directory')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                       help='Path to save trained models')
    
    # Task selection
    parser.add_argument('--task', type=str, choices=['MI', 'SSVEP', 'both'], default='both',
                       help='Which task to train')
    
    # Model configuration
    parser.add_argument('--mi_model', type=str, default='EEGNet',
                       choices=['LDA', 'SVM', 'RandomForest', 'EEGNet', 'DeepConvNet', 
                               'EnhancedFeatureClassifier', 'BiLSTMClassifier'],
                       help='Model type for MI task')
    parser.add_argument('--ssvep_model', type=str, default='EnhancedFeatureClassifier',
                       choices=['LDA', 'SVM', 'RandomForest', 'EEGNet', 'DeepConvNet',
                               'EnhancedFeatureClassifier', 'SSVEPFormer', 'BiLSTMClassifier'],
                       help='Model type for SSVEP task')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for training')
    
    # Model-specific parameters
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for LSTM models')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension for Transformer')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Training configuration
    train_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patience': args.patience,
        'device': args.device
    }
    
    # Model configurations
    model_configs = {
        'MI': {
            'dropout': args.dropout,
            'hidden_dim': args.hidden_dim,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers
        },
        'SSVEP': {
            'dropout': args.dropout,
            'hidden_dim': args.hidden_dim,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers
        }
    }
    
    # Load and train models
    if args.task in ['MI', 'both']:
        try:
            with open(os.path.join(args.data_path, 'mi_data.pkl'), 'rb') as f:
                mi_data = pickle.load(f)
            
            mi_model, mi_metrics = train_task_model(
                mi_data, 'MI', args.mi_model, 
                model_configs['MI'], train_config, args.save_path
            )
            
            print(f"✓ MI model training completed")
            print(f"  Final metrics: {mi_metrics}")
            
        except FileNotFoundError:
            print("⚠ MI data not found. Skipping MI training.")
    
    if args.task in ['SSVEP', 'both']:
        try:
            with open(os.path.join(args.data_path, 'ssvep_data.pkl'), 'rb') as f:
                ssvep_data = pickle.load(f)
            
            ssvep_model, ssvep_metrics = train_task_model(
                ssvep_data, 'SSVEP', args.ssvep_model,
                model_configs['SSVEP'], train_config, args.save_path
            )
            
            print(f"✓ SSVEP model training completed")
            print(f"  Final metrics: {ssvep_metrics}")
            
        except FileNotFoundError:
            print("⚠ SSVEP data not found. Skipping SSVEP training.")
    
    print(f"\n✓ Training completed! Models saved to {args.save_path}")


if __name__ == "__main__":
    main()
