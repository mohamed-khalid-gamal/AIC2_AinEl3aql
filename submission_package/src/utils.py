"""
Utility functions for EEG data processing and model operations.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, Tuple, List


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration dictionary to JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, path: str) -> None:
    """Save object to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load object from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: float, precision: int = 4) -> str:
    """Format number with appropriate precision."""
    if num == 0:
        return "0"
    elif abs(num) >= 1000:
        return f"{num:,.0f}"
    elif abs(num) >= 1:
        return f"{num:.{precision}f}"
    else:
        return f"{num:.{precision}f}"


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]) -> None:
    """Print a summary of the model architecture."""
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Try to get output size
    try:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            output = model(dummy_input)
            print(f"Output size: {output.shape[1:]}")
    except Exception as e:
        print(f"Could not determine output size: {e}")


def validate_data_split_ids(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame) -> bool:
    """Validate that data split IDs don't overlap."""
    train_ids = set(train_df['id'])
    val_ids = set(val_df['id'])
    test_ids = set(test_df['id'])
    
    # Check for overlaps
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    if train_val_overlap:
        print(f"Warning: Train-Validation overlap: {len(train_val_overlap)} samples")
        return False
    
    if train_test_overlap:
        print(f"Warning: Train-Test overlap: {len(train_test_overlap)} samples")
        return False
    
    if val_test_overlap:
        print(f"Warning: Validation-Test overlap: {len(val_test_overlap)} samples")
        return False
    
    print("✓ Data split validation passed - no ID overlaps found")
    return True


def check_data_consistency(data_dict: Dict[str, Any]) -> None:
    """Check consistency of preprocessed data."""
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    
    # Check shapes
    assert X_train.shape[0] == len(y_train), "Train features and labels size mismatch"
    assert X_val.shape[0] == len(y_val), "Validation features and labels size mismatch"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature dimension mismatch"
    
    # Check for NaN values
    assert not np.isnan(X_train).any(), "NaN values found in training features"
    assert not np.isnan(X_val).any(), "NaN values found in validation features"
    assert not np.isnan(X_test).any(), "NaN values found in test features"
    
    # Check label ranges
    assert y_train.min() >= 0, "Negative labels found in training set"
    assert y_val.min() >= 0, "Negative labels found in validation set"
    assert y_train.max() == y_val.max(), "Label range mismatch between train and validation"
    
    print("✓ Data consistency check passed")


def create_submission_template(test_ids: List[int], output_path: str) -> None:
    """Create a template submission file."""
    submission = pd.DataFrame({
        'id': sorted(test_ids),
        'label': 0  # Default prediction
    })
    submission.to_csv(output_path, index=False)
    print(f"Template submission created: {output_path}")


def merge_predictions(pred_dfs: List[pd.DataFrame], output_path: str) -> pd.DataFrame:
    """Merge predictions from multiple tasks."""
    combined = pd.concat(pred_dfs, ignore_index=True)
    combined = combined.sort_values('id').reset_index(drop=True)
    
    # Check for missing IDs
    expected_ids = set(range(combined['id'].min(), combined['id'].max() + 1))
    actual_ids = set(combined['id'])
    missing_ids = expected_ids - actual_ids
    
    if missing_ids:
        print(f"Warning: Missing predictions for {len(missing_ids)} IDs")
    
    # Save combined predictions
    submission = combined[['id', 'label']]
    submission.to_csv(output_path, index=False)
    
    return combined


def log_metrics(metrics: Dict[str, float], log_file: str) -> None:
    """Log metrics to file with timestamp."""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}]\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {format_number(value)}\n")


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
                
        return False


class MetricsTracker:
    """Track training metrics over time."""
    
    def __init__(self):
        self.metrics = {}
        
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for a metric."""
        return self.metrics.get(key, [None])[-1]
    
    def get_best(self, key: str, mode: str = 'max') -> Optional[float]:
        """Get best value for a metric."""
        values = self.metrics.get(key, [])
        if not values:
            return None
        return max(values) if mode == 'max' else min(values)
    
    def save(self, path: str):
        """Save metrics to file."""
        save_pickle(self.metrics, path)
    
    def load(self, path: str):
        """Load metrics from file."""
        self.metrics = load_pickle(path)


def verify_submission_format(submission_path: str, expected_ids: List[int]) -> bool:
    """Verify submission file format."""
    try:
        df = pd.read_csv(submission_path)
        
        # Check columns
        if list(df.columns) != ['id', 'label']:
            print(f"Error: Expected columns ['id', 'label'], got {list(df.columns)}")
            return False
        
        # Check ID completeness
        missing_ids = set(expected_ids) - set(df['id'])
        if missing_ids:
            print(f"Error: Missing {len(missing_ids)} IDs in submission")
            return False
        
        # Check for extra IDs
        extra_ids = set(df['id']) - set(expected_ids)
        if extra_ids:
            print(f"Warning: {len(extra_ids)} extra IDs in submission")
        
        # Check for NaN values
        if df.isna().any().any():
            print("Error: NaN values found in submission")
            return False
        
        print("✓ Submission format validation passed")
        return True
        
    except Exception as e:
        print(f"Error reading submission file: {e}")
        return False


def print_data_info(data_dict: Dict[str, Any], task_name: str) -> None:
    """Print information about loaded data."""
    print(f"\n{task_name} Data Information:")
    print(f"  Training samples: {len(data_dict['X_train'])}")
    print(f"  Validation samples: {len(data_dict['X_val'])}")
    print(f"  Test samples: {len(data_dict['X_test'])}")
    print(f"  Features: {data_dict['n_features']}")
    print(f"  Classes: {data_dict['n_classes']}")
    print(f"  Feature type: {data_dict['feature_type']}")
    
    # Class distribution
    unique, counts = np.unique(data_dict['y_train'], return_counts=True)
    print(f"  Class distribution (train): {dict(zip(unique, counts))}")


# Data augmentation functions for EEG
def add_noise(data: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to EEG data."""
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise


def time_shift(data: np.ndarray, shift_range: int = 10) -> np.ndarray:
    """Apply random time shift to EEG data."""
    shift = np.random.randint(-shift_range, shift_range + 1)
    if shift > 0:
        return np.concatenate([data[shift:], data[:shift]], axis=0)
    elif shift < 0:
        return np.concatenate([data[shift:], data[:shift]], axis=0)
    return data


def amplitude_scale(data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Apply random amplitude scaling to EEG data."""
    scale = np.random.uniform(*scale_range)
    return data * scale
