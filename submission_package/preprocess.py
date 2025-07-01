#!/usr/bin/env python3
"""
EEG Data Preprocessing Script
Handles filtering, artifact removal, feature extraction, and data splitting for both MI and SSVEP tasks.
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, stft, welch, hilbert, gausspulse, chirp, spectrogram
from scipy.stats import linregress
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_decomposition import CCA
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from tqdm import tqdm
except ImportError:
    # Fallback progress bar
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import MNE components (with fallback)
try:
    from mne.decoding import CSP
    from mne.preprocessing import ICA as mne_ICA
    MNE_AVAILABLE = True
except ImportError:
    print("Warning: MNE not available. CSP and ICA features will be disabled.")
    MNE_AVAILABLE = False
    
    # Fallback CSP implementation
    class CSP:
        def __init__(self, n_components=4, reg=None, log=True):
            self.n_components = n_components
            self.reg = reg
            self.log = log
            print("Using fallback CSP implementation")
            
        def fit(self, X, y):
            # Basic CSP implementation using eigendecomposition
            from sklearn.covariance import EmpiricalCovariance
            
            # Calculate covariance matrices for each class
            classes = np.unique(y)
            if len(classes) != 2:
                raise ValueError("CSP requires exactly 2 classes")
                
            X1 = X[y == classes[0]]
            X2 = X[y == classes[1]]
            
            # Calculate average covariance matrices
            C1 = np.mean([np.cov(trial.T) for trial in X1], axis=0)
            C2 = np.mean([np.cov(trial.T) for trial in X2], axis=0)
            
            # Solve generalized eigenvalue problem
            eigenvals, eigenvecs = eigh(C1, C1 + C2)
            
            # Sort and select components
            ix = np.argsort(eigenvals)
            self.filters_ = eigenvecs[:, ix]
            
            # Select top and bottom components
            n_comp_half = self.n_components // 2
            selected_indices = np.concatenate([
                ix[:n_comp_half],  # Bottom components
                ix[-n_comp_half:]  # Top components
            ])
            self.filters_ = eigenvecs[:, selected_indices]
            
            return self
            
        def transform(self, X):
            # Apply spatial filters
            X_filtered = np.zeros((X.shape[0], self.n_components, X.shape[2]))
            for i, trial in enumerate(X):
                X_filtered[i] = self.filters_.T @ trial.T
                
            if self.log:
                # Calculate log variance
                features = np.log(np.var(X_filtered, axis=2))
            else:
                features = np.var(X_filtered, axis=2)
                
            return features

# Try to import PyWavelets
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    print("Warning: PyWavelets not available. Using scipy-based alternatives.")
    PYWT_AVAILABLE = False


class EEGPreprocessor(BaseEstimator, TransformerMixin):
    """Basic EEG preprocessing including filtering and normalization."""
    
    def __init__(self, filter_low=8, filter_high=30, sampling_rate=250):
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.sampling_rate = sampling_rate
        self.selected_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transform list of DataFrames to normalized epochs."""
        all_epochs = []
        
        for df in X:
            # Channel selection
            eeg_data = df[self.selected_channels].values
            
            # Band-pass filter
            nyquist = 0.5 * self.sampling_rate
            low = self.filter_low / nyquist
            high = self.filter_high / nyquist
            b, a = butter(5, [low, high], btype='band')
            filtered = filtfilt(b, a, eeg_data, axis=0)
            
            # Per-channel normalization
            normalized = (filtered - filtered.mean(axis=0)) / (filtered.std(axis=0) + 1e-8)
            all_epochs.append(normalized)
                
        return np.array(all_epochs)


class CSPFeatures(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns feature extraction."""
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        if MNE_AVAILABLE:
            self.csp = CSP(n_components=n_components, reg=None, log=True)
        else:
            raise ImportError("MNE is required for CSP features")
        
    def fit(self, X, y):
        # X shape: (n_trials, time_points, channels)
        X_csp = X.transpose(0, 2, 1)  # MNE expects (trials, channels, time)
        self.csp.fit(X_csp, y)
        return self
        
    def transform(self, X):
        X_csp = X.transpose(0, 2, 1)
        return self.csp.transform(X_csp)


class FBCSPFeatures(BaseEstimator, TransformerMixin):
    """Filter Bank Common Spatial Patterns."""
    
    def __init__(self, n_components=4, freq_bands=None, sampling_rate=250):
        self.n_components = n_components
        self.sampling_rate = sampling_rate
        if freq_bands is None:
            self.freq_bands = [(8, 12), (12, 16), (16, 24), (24, 30)]
        else:
            self.freq_bands = freq_bands
        self.csp_models = []
        
    def fit(self, X, y):
        if not MNE_AVAILABLE:
            raise ImportError("MNE is required for FBCSP features")
            
        self.csp_models = []
        for low, high in self.freq_bands:
            filtered = self._bandpass_filter(X, low, high)
            csp = CSP(n_components=self.n_components, reg=None, log=True)
            csp.fit(filtered.transpose(0, 2, 1), y)
            self.csp_models.append((low, high, csp))
        return self
        
    def transform(self, X):
        features = []
        for low, high, csp in self.csp_models:
            filtered = self._bandpass_filter(X, low, high)
            csp_feats = csp.transform(filtered.transpose(0, 2, 1))
            features.append(csp_feats)
        return np.concatenate(features, axis=1)
    
    def _bandpass_filter(self, X, low, high):
        nyquist = 0.5 * self.sampling_rate
        low_norm = low / nyquist
        high_norm = high / nyquist
        b, a = butter(5, [low_norm, high_norm], btype='band')
        
        filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                filtered[i, :, j] = filtfilt(b, a, X[i, :, j])
        return filtered


class FBCCAExtractor:
    """Filter Bank Canonical Correlation Analysis for SSVEP."""
    
    def __init__(self, fs=250, num_harmonics=2, num_subbands=5):
        self.fs = fs
        self.num_harmonics = num_harmonics
        self.num_subbands = num_subbands
        self.target_freqs = [7, 8, 10, 13]  # SSVEP targets
        self.subbands = [
            (5, 40), (6, 38), (7, 36), (8, 34), (9, 32)
        ][:num_subbands]

    def _bandpass_filter(self, data, low_freq, high_freq, order=4):
        nyquist = 0.5 * self.fs
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    def _generate_reference_signals(self, freq, n_samples):
        t = np.arange(n_samples) / self.fs
        ref = [
            np.sin(2 * np.pi * freq * i * t) for i in range(1, self.num_harmonics+1)
        ] + [
            np.cos(2 * np.pi * freq * i * t) for i in range(1, self.num_harmonics+1)
        ]
        return np.stack(ref, axis=1)

    def _cca_correlation(self, X, Y):
        cca = CCA(n_components=1)
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        return np.corrcoef(X_c.T, Y_c.T)[0, 1]

    def extract_fbcca_features(self, eeg_data):
        n_samples = eeg_data.shape[0]
        corrs = []

        for low, high in self.subbands:
            filtered = self._bandpass_filter(eeg_data, low, high)
            sub_corrs = [
                self._cca_correlation(filtered, self._generate_reference_signals(freq, n_samples))
                for freq in self.target_freqs
            ]
            corrs.append(sub_corrs)

        corrs = np.array(corrs)
        weights = 1 / np.arange(1, self.num_subbands + 1)
        weights /= weights.sum()
        return np.dot(weights, corrs)


class STFTFeatures(BaseEstimator, TransformerMixin):
    """Short-Time Fourier Transform features."""
    
    def __init__(self, nperseg=250, noverlap=125, sampling_rate=250):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.sampling_rate = sampling_rate
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        for trial in X:
            trial_features = []
            for channel in range(trial.shape[1]):
                f, t, Zxx = stft(trial[:, channel], 
                                fs=self.sampling_rate,
                                nperseg=self.nperseg,
                                noverlap=self.noverlap)
                psd = np.abs(Zxx) ** 2
                
                # Extract alpha (8-12Hz) and beta (12-30Hz) bands
                alpha = psd[(f >= 8) & (f < 12)].mean()
                beta = psd[(f >= 12) & (f <= 30)].mean()
                trial_features.extend([alpha, beta])
                
            features.append(trial_features)
        return np.array(features)


class HiguchiFDFeatures(BaseEstimator, TransformerMixin):
    """Higuchi Fractal Dimension features."""
    
    def __init__(self, kmax=10):
        self.kmax = kmax
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        for trial in X:
            trial_features = []
            for channel in range(trial.shape[1]):
                hfd = self._higuchi_fd(trial[:, channel], self.kmax)
                trial_features.append(hfd)
            features.append(trial_features)
        return np.array(features)
    
    def _higuchi_fd(self, x, kmax=10):
        """Compute Higuchi Fractal Dimension of a time series."""
        n = len(x)
        lk = np.zeros(kmax)
        x = np.asarray(x)
        
        for k in range(1, kmax+1):
            lm = np.zeros((k,))
            for m in range(k):
                ll = 0
                max_i = int(np.floor((n - m - 1) / k))
                for i in range(1, max_i):
                    ll += abs(x[m + i*k] - x[m + (i-1)*k])
                ll = ll * (n - 1) / (max_i * k)
                lm[m] = np.log(ll / k)
            lk[k-1] = np.mean(lm)
        
        hfd = np.polyfit(np.log(range(1, kmax+1)), lk, 1)[0]
        return hfd


def load_index_csvs(base_path):
    """Load and prepare index CSV files with label encoding."""
    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

    # Create label encoders for both tasks
    le_mi = LabelEncoder()
    le_ssvep = LabelEncoder()

    # Fit encoders on training data
    if 'label' in train_df.columns:
        mi_train_labels = train_df[train_df['task'] == 'MI']['label']
        ssvep_train_labels = train_df[train_df['task'] == 'SSVEP']['label']
        
        if len(mi_train_labels) > 0:
            le_mi.fit(mi_train_labels)
            # Transform MI labels in all splits
            for df in [train_df, validation_df]:
                if 'label' in df.columns:
                    mi_mask = df['task'] == 'MI'
                    if mi_mask.any():
                        df.loc[mi_mask, 'label'] = le_mi.transform(df.loc[mi_mask, 'label'])
        
        if len(ssvep_train_labels) > 0:
            le_ssvep.fit(ssvep_train_labels)
            # Transform SSVEP labels in all splits
            for df in [train_df, validation_df]:
                if 'label' in df.columns:
                    ssvep_mask = df['task'] == 'SSVEP'
                    if ssvep_mask.any():
                        df.loc[ssvep_mask, 'label'] = le_ssvep.transform(df.loc[ssvep_mask, 'label'])

    return train_df, validation_df, test_df, le_mi, le_ssvep


def load_raw_eeg(df, base_path, task='MI'):
    """Load raw EEG data from CSV files."""
    raws = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {task} data"):
        subject = str(row['subject_id'])
        session = str(row['trial_session'])
        
        # Determine dataset split
        if row['id'] <= 4800:
            split = 'train'
        elif row['id'] <= 4900:
            split = 'validation'
        else:
            split = 'test'
            
        fpath = os.path.join(base_path, task, split, subject, session, 'EEGdata.csv')
        
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")
            
        eeg = pd.read_csv(fpath)

        # Extract correct trial slice
        trial = int(row['trial'])
        samples_per_trial = 2250 if task == 'MI' else 1750
        start = (trial - 1) * samples_per_trial
        end = start + samples_per_trial
        raws.append(eeg.iloc[start:end])
    return raws


def extract_features_for_task(df, base_path, task='MI', feature_type='CSP', 
                             feature_params=None, preprocess_params=None,
                             fitted_extractor=None):
    """Extract features for a specific task."""
    if feature_params is None:
        feature_params = {}
    if preprocess_params is None:
        preprocess_params = {}
    
    # Filter dataframe for specific task
    task_df = df[df['task'] == task].copy().reset_index(drop=True)
    
    if len(task_df) == 0:
        print(f"No {task} samples found in dataframe")
        return None, None, None, None
    
    # Load raw EEG data
    print(f"Loading {task} raw EEG data...")
    X_raw = load_raw_eeg(task_df, base_path, task)
    
    # Get labels if available
    y = task_df['label'].values if 'label' in task_df.columns else None
    ids = task_df['id'].values
    
    # Preprocessing
    print(f"Preprocessing {task} data...")
    preprocessor = EEGPreprocessor(**preprocess_params)
    X_epochs = preprocessor.fit_transform(X_raw)
    
    # Feature extraction
    print(f"Extracting {feature_type} features for {task}...")
    
    if task == 'SSVEP' and feature_type == 'FBCCA':
        # Special handling for SSVEP FBCCA features
        eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
        fbcca_extractor = FBCCAExtractor(**feature_params)
        
        features = []
        for trial_data in X_raw:
            available_channels = [ch for ch in eeg_channels if ch in trial_data.columns]
            eeg_data = trial_data[available_channels].values
            try:
                fbcca_feats = fbcca_extractor.extract_fbcca_features(eeg_data)
            except Exception as e:
                print(f"FBCCA failed: {e}")
                fbcca_feats = np.zeros(len(fbcca_extractor.target_freqs))
            features.append(fbcca_feats)
        X_features = np.array(features)
        feat_extractor = None
        
    elif feature_type == 'CSP':
        if fitted_extractor is not None:
            # Use pre-fitted extractor for test data
            feat_extractor = fitted_extractor
            X_features = feat_extractor.transform(X_epochs)
        else:
            # Fit new extractor for training data
            if y is None:
                raise ValueError("CSP requires labels for fitting")
            feat_extractor = CSPFeatures(**feature_params)
            X_features = feat_extractor.fit_transform(X_epochs, y)
        
    elif feature_type == 'FBCSP':
        if fitted_extractor is not None:
            # Use pre-fitted extractor for test data
            feat_extractor = fitted_extractor
            X_features = feat_extractor.transform(X_epochs)
        else:
            # Fit new extractor for training data
            if y is None:
                raise ValueError("FBCSP requires labels for fitting")
            feat_extractor = FBCSPFeatures(**feature_params)
            X_features = feat_extractor.fit_transform(X_epochs, y)
        
    elif feature_type == 'STFT':
        feat_extractor = STFTFeatures(**feature_params)
        X_features = feat_extractor.fit_transform(X_epochs)
        
    elif feature_type == 'HiguchiFD':
        feat_extractor = HiguchiFDFeatures(**feature_params)
        X_features = feat_extractor.fit_transform(X_epochs)
        
    elif feature_type == 'RAW':
        X_features = X_epochs.reshape(X_epochs.shape[0], -1)
        feat_extractor = None
        
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    print(f"Extracted {X_features.shape[1]} features from {X_features.shape[0]} samples")
    return X_features, y, ids, feat_extractor


def preprocess_and_save(args):
    """Main preprocessing function."""
    print("="*50)
    print("EEG DATA PREPROCESSING")
    print("="*50)
    
    # Load index files
    print("Loading index files...")
    train_df, val_df, test_df, le_mi, le_ssvep = load_index_csvs(args.data_path)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Save label encoders
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump({'mi': le_mi, 'ssvep': le_ssvep}, f)
    
    # Process each task
    tasks_config = {
        'MI': {
            'feature_type': args.mi_feature_type,
            'feature_params': {'n_components': args.mi_csp_components},
            'preprocess_params': {
                'filter_low': args.mi_filter_low,
                'filter_high': args.mi_filter_high
            }
        },
        'SSVEP': {
            'feature_type': args.ssvep_feature_type,
            'feature_params': {
                'num_harmonics': args.ssvep_harmonics,
                'num_subbands': args.ssvep_subbands
            },
            'preprocess_params': {
                'filter_low': args.ssvep_filter_low,
                'filter_high': args.ssvep_filter_high
            }
        }
    }
    
    for task, config in tasks_config.items():
        print(f"\n{'='*30}")
        print(f"Processing {task} Task")
        print(f"{'='*30}")
        
        # Process training data
        X_train, y_train, ids_train, feat_extractor = extract_features_for_task(
            train_df, args.data_path, task, 
            config['feature_type'], 
            config['feature_params'],
            config['preprocess_params']
        )
        
        if X_train is not None:
            # Process validation data
            X_val, y_val, ids_val, _ = extract_features_for_task(
                val_df, args.data_path, task,
                config['feature_type'],
                config['feature_params'],
                config['preprocess_params'],
                fitted_extractor=feat_extractor
            )
            
            # Process test data
            X_test, y_test, ids_test, _ = extract_features_for_task(
                test_df, args.data_path, task,
                config['feature_type'],
                config['feature_params'],
                config['preprocess_params'],
                fitted_extractor=feat_extractor
            )
            
            # Save feature extractor if needed
            if feat_extractor is not None:
                with open(os.path.join(args.output_path, f'{task.lower()}_feature_extractor.pkl'), 'wb') as f:
                    pickle.dump(feat_extractor, f)
            
            # Feature scaling and selection
            if args.scale_features:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                # Save scaler
                with open(os.path.join(args.output_path, f'{task.lower()}_scaler.pkl'), 'wb') as f:
                    pickle.dump(scaler, f)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
                X_test_scaled = X_test
            
            if args.select_features > 0:
                k = min(args.select_features, X_train_scaled.shape[1])
                selector = SelectKBest(f_classif, k=k)
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                X_val_selected = selector.transform(X_val_scaled)
                X_test_selected = selector.transform(X_test_scaled)
                
                # Save selector
                with open(os.path.join(args.output_path, f'{task.lower()}_selector.pkl'), 'wb') as f:
                    pickle.dump(selector, f)
                    
                print(f"Selected {X_train_selected.shape[1]} features out of {X_train_scaled.shape[1]}")
            else:
                X_train_selected = X_train_scaled
                X_val_selected = X_val_scaled
                X_test_selected = X_test_scaled
            
            # Save processed data
            task_data = {
                'X_train': X_train_selected,
                'y_train': y_train,
                'ids_train': ids_train,
                'X_val': X_val_selected,
                'y_val': y_val,
                'ids_val': ids_val,
                'X_test': X_test_selected,
                'y_test': y_test,
                'ids_test': ids_test,
                'feature_type': config['feature_type'],
                'n_features': X_train_selected.shape[1],
                'n_classes': len(np.unique(y_train)) if y_train is not None else None
            }
            
            with open(os.path.join(args.output_path, f'{task.lower()}_data.pkl'), 'wb') as f:
                pickle.dump(task_data, f)
            
            print(f"✓ Saved {task} processed data")
            print(f"  Features: {X_train_selected.shape[1]}")
            print(f"  Classes: {len(np.unique(y_train)) if y_train is not None else 'N/A'}")
            print(f"  Train samples: {len(X_train_selected)}")
            print(f"  Val samples: {len(X_val_selected)}")
            print(f"  Test samples: {len(X_test_selected)}")
    
    print(f"\n✓ Preprocessing complete! Data saved to {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description='EEG Data Preprocessing')
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the dataset directory')
    parser.add_argument('--output_path', type=str, default='./preprocessed_data',
                       help='Path to save preprocessed data')
    
    # MI task parameters
    parser.add_argument('--mi_feature_type', type=str, default='FBCSP',
                       choices=['CSP', 'FBCSP', 'STFT', 'HiguchiFD', 'RAW'],
                       help='Feature type for MI task')
    parser.add_argument('--mi_csp_components', type=int, default=4,
                       help='Number of CSP components for MI')
    parser.add_argument('--mi_filter_low', type=float, default=8.0,
                       help='Low cutoff frequency for MI (Hz)')
    parser.add_argument('--mi_filter_high', type=float, default=30.0,
                       help='High cutoff frequency for MI (Hz)')
    
    # SSVEP task parameters
    parser.add_argument('--ssvep_feature_type', type=str, default='FBCCA',
                       choices=['FBCCA', 'STFT', 'RAW'],
                       help='Feature type for SSVEP task')
    parser.add_argument('--ssvep_harmonics', type=int, default=2,
                       help='Number of harmonics for SSVEP')
    parser.add_argument('--ssvep_subbands', type=int, default=5,
                       help='Number of subbands for SSVEP')
    parser.add_argument('--ssvep_filter_low', type=float, default=5.0,
                       help='Low cutoff frequency for SSVEP (Hz)')
    parser.add_argument('--ssvep_filter_high', type=float, default=40.0,
                       help='High cutoff frequency for SSVEP (Hz)')
    
    # Processing options
    parser.add_argument('--scale_features', action='store_true',
                       help='Apply feature scaling')
    parser.add_argument('--select_features', type=int, default=0,
                       help='Number of features to select (0 = no selection)')
    
    args = parser.parse_args()
    preprocess_and_save(args)


if __name__ == "__main__":
    main()
