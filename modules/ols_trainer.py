import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util_scripts'))
import sequence_utils as utils

try:
    import cudf
    import cuml
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class OLSTrainer:
    def __init__(self, config):
        self.config = config
    
    def train(self, predictions_csv, output_name, kmer_size=6, num=None, seed=42, mode=3):
        """Train OLS model on k-mer features"""
        
        if not os.path.exists(predictions_csv):
            raise FileNotFoundError(f"Predictions file not found: {predictions_csv}")
        
        df = pd.read_csv(predictions_csv)
        
        if 'sequence' not in df.columns or 'score' not in df.columns:
            raise ValueError(f"Predictions CSV must contain 'sequence' and 'score' columns. Found columns: {df.columns.tolist()}")
        
        print(f"Loaded {len(df)} predictions from {predictions_csv}")
        
        # Apply sequence limits based on k-mer size
        if num is None:
            max_sequences = self.config['ols_training']['max_sequences']
            if isinstance(max_sequences, dict):
                num = max_sequences.get(kmer_size, 1800000)
            else:
                num = max_sequences
        
        if df.shape[0] > num:
            print(f"Sampling {num} sequences from {df.shape[0]} total sequences")
            df = df.sample(n=num, random_state=seed)
        
        sequences = df['sequence'].values
        scores = df['score'].values
        
        # Extract k-mer features
        print(f"Extracting {kmer_size}-mers...")
        X, colnames = utils.extract_kmers(sequences, kmer_size)
        y = scores
        
        # Train OLS model
        try:
            if CUPY_AVAILABLE:
                lm_dic, cov_matrix = self._run_cupy(X, y, colnames, cov=(mode in [2, 3]))
            else:
                lm_dic, cov_matrix = self._run_numpy(X, y, colnames, cov=(mode in [2, 3]))
        except Exception as e:
            print(f"Training failed: {e}")
            return None
        
        # Save outputs
        self._save_outputs(lm_dic, cov_matrix, output_name, kmer_size, mode)
        
        return lm_dic
    
    def _run_cupy(self, X, y, colnames, cov=False):
        """Run OLS using CuPy"""
        X = np.float32(X)
        y = np.float32(y)
        
        X = cp.array(X)
        y = cp.array(y)
        
        # Compute OLS estimator
        XTX_inv = cp.linalg.inv(X.T.dot(X))
        beta = XTX_inv.dot(X.T).dot(y)
        beta_np = beta.get()
        
        cov_matrix = None
        if cov:
            # Compute residuals and covariance
            residuals = y - X.dot(beta)
            n = X.shape[0]
            k = X.shape[1]
            sigma_squared = cp.sum(residuals ** 2) / (n - k)
            cov_matrix = sigma_squared * XTX_inv
            cov_matrix = cov_matrix.get()
        
        return dict(zip(colnames, beta_np)), cov_matrix
    
    def _run_numpy(self, X, y, colnames, cov=False):
        """Run OLS using NumPy (fallback)"""
        X = np.float32(X)
        y = np.float32(y)
        
        # Compute OLS estimator
        XTX_inv = np.linalg.inv(X.T.dot(X))
        beta = XTX_inv.dot(X.T).dot(y)
        
        cov_matrix = None
        if cov:
            # Compute residuals and covariance
            residuals = y - X.dot(beta)
            n = X.shape[0]
            k = X.shape[1]
            sigma_squared = np.sum(residuals ** 2) / (n - k)
            cov_matrix = sigma_squared * XTX_inv
        
        return dict(zip(colnames, beta)), cov_matrix
    
    def _save_outputs(self, lm_dic, cov_matrix, output_name, kmer_size, mode):
        """Save OLS outputs based on mode"""
        
        # Save coefficients for modes 1 and 3
        if mode in [1, 3]:
            os.makedirs('k_mer_weights', exist_ok=True)
            weights_filename = f'k_mer_weights/{output_name}_{kmer_size}mer.qbic'
            pickle.dump(lm_dic, open(weights_filename, 'wb'))
        
        # Save covariance for modes 2 and 3
        if mode in [2, 3] and cov_matrix is not None:
            os.makedirs('covariance_matrices', exist_ok=True)
            self._save_symm_npy_half(cov_matrix, f'covariance_matrices/{output_name}_{kmer_size}mer')
    
    def _save_symm_npy_half(self, cov_matrix, output_path):
        """Save only upper triangle of symmetric matrix"""
        n, m = cov_matrix.shape
        assert n == m, "Matrix must be square."
        
        iu, ju = np.triu_indices(n)
        upper_half = cov_matrix[iu, ju].copy().astype(np.float32)
        
        output_path_with_size = f"{output_path}.cov_{n}.npy"
        np.save(output_path_with_size, upper_half)