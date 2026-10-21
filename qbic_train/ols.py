"""k-mer OLS module (Methods: "Interpretable k-mer OLS model"; "Statistical
inference for variant effects").

beta = (X'X)^-1 X'y and Sigma = sigma^2 (X'X)^-1, with sigma^2 = RSS / (n - d),
computed in float32 on the GPU with CuPy when available, else with NumPy.
"""

import pickle

import numpy as np
import pandas as pd

from .kmers import canonical_kmers, kmer_count_matrix


def gpu_available() -> bool:
    try:
        import cupy as cp
        cp.zeros(1).sum()
        return True
    except Exception:
        return False


def sample_for_ols(scores: pd.DataFrame, k: int, max_sequences: dict, seed: int) -> pd.DataFrame:
    """Cap the scored reads used for a k-mer model (e.g. 450,000 for 7-mers)."""
    cap = max_sequences.get(k)
    if cap is not None and scores.shape[0] > cap:
        scores = scores.sample(n=cap, random_state=seed)
    return scores


def fit_ols(X, y, use_gpu: bool):
    """Return ``(beta, cov)`` as float32 NumPy arrays."""
    X = np.float32(X)
    y = np.float32(y)
    xp = np
    if use_gpu:
        import cupy as xp
        X, y = xp.array(X), xp.array(y)
    XTX_inv = xp.linalg.inv(X.T.dot(X))
    beta = XTX_inv.dot(X.T).dot(y)
    residuals = y - X.dot(beta)
    n, d = X.shape
    sigma_squared = xp.sum(residuals ** 2) / (n - d)
    cov = sigma_squared * XTX_inv
    if use_gpu:
        beta, cov = beta.get(), cov.get()
    return beta, cov


def train_kmer_model(scores: pd.DataFrame, k: int, use_gpu: bool):
    """Fit one k-mer model on scored reads (``sequence``, ``score`` columns)."""
    d = len(canonical_kmers(k))
    if scores.shape[0] <= d:
        raise ValueError(f"{k}-mer OLS needs more scored reads ({scores.shape[0]}) than k-mer features ({d})")
    X = kmer_count_matrix(scores["sequence"].values, k)
    beta, cov = fit_ols(X, scores["score"].values, use_gpu)
    if not np.all(np.isfinite(beta)):
        raise ValueError(f"{k}-mer OLS gave non-finite coefficients (X'X singular?)")
    return dict(zip(canonical_kmers(k), beta, strict=True)), cov


def save_weights(path, weights: dict) -> None:
    """``*.weights.qbic``: pickled dict canonical k-mer -> float32, as qbic-selex reads it."""
    with open(path, "wb") as f:
        pickle.dump(weights, f)


def save_covariance(path_prefix, cov: np.ndarray) -> str:
    """Write the upper triangle (np.triu_indices order) as ``<prefix>.cov_<n>.npy``."""
    n = cov.shape[0]
    iu, ju = np.triu_indices(n)
    path = f"{path_prefix}.cov_{n}.npy"
    np.save(path, cov[iu, ju].copy().astype(np.float32))
    return path
