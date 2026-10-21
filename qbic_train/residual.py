"""TF-specific residual model (Methods: "TF-specific residual model architecture
and training"): train on early- vs late-cycle reads with the frozen universal
bias model, then score every read and select the TF binding score f_QBiC."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .networks import ResidualModel, SequenceDataset


def train_residual(bias_model, early_reads, late_reads, cfg, device, log=print):
    """Train one residual model; late-cycle reads are the positive class.

    The steps that draw random numbers run in the original order (dataset,
    shuffled loader, model initialisation, one pass over the shuffled loader for
    the class weight, then the epochs), so a seeded run reproduces the original.
    Returns ``(model, dataset, sequences, final_loss, final_auroc)``.
    """
    if not early_reads or not late_reads:
        raise ValueError("both the early and the late cycle need at least one read")
    sequences = early_reads + late_reads
    X = np.array(sequences)
    y = np.concatenate([np.zeros(len(early_reads), dtype=np.float32), np.ones(len(late_reads), dtype=np.float32)])

    dataset = SequenceDataset(X, y)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    model = ResidualModel(bias_model, num_filters=cfg["num_filters"], kernel_size=cfg["kernel_size"]).to(device)

    y_all = torch.cat([y_batch for _, y_batch in loader])
    pos_weight = torch.tensor([(1 - y_all.mean()) / y_all.mean()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], eps=cfg["adam_eps"], amsgrad=cfg["amsgrad"])

    model.train()
    for epoch in range(1, cfg["num_epochs"] + 1):
        total_loss, preds, labels = 0.0, [], []
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_norm"])
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            with torch.no_grad():
                preds.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(y_batch.cpu().numpy())
        epoch_loss = total_loss / len(dataset)
        epoch_auroc = roc_auc_score(np.concatenate(labels), np.concatenate(preds))
        log(f"    epoch {epoch}/{cfg['num_epochs']}  loss {epoch_loss:.4f}  AUROC {epoch_auroc:.4f}")
    return model, dataset, sequences, epoch_loss, epoch_auroc


def select_scores(combined: pd.DataFrame, residual: pd.DataFrame, alpha: float):
    """f_QBiC = f_RES if alpha >= 0, else f_SELEX. Returns ``(selected, alternate)``."""
    return (residual, combined) if alpha >= 0 else (combined, residual)


def _unique_capped(sequences, scores, max_sequences, seed):
    df = pd.DataFrame({"sequence": sequences, "score": scores}).drop_duplicates(subset=["sequence"], keep="first")
    if df.shape[0] > max_sequences:
        df = df.sample(n=max_sequences, random_state=seed)
    return df


def score_reads(model, dataset, sequences, cfg, device, seed):
    """Score every read with f_SELEX and f_RES; keep one row per unique read
    (at most ``max_scored_sequences``, sampled with ``seed``) and select f_QBiC.

    Returns ``(selected, alternate, alpha)``.
    """
    model.eval()
    combined, residual = [], []
    with torch.no_grad():
        for X_batch, _ in DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False):
            tf_logit, res_logit = model(X_batch.to(device), return_residual=True)
            combined.append(tf_logit.squeeze(-1).cpu().numpy())
            residual.append(res_logit.squeeze(-1).cpu().numpy())
    alpha = model.alpha
    cap = cfg["max_scored_sequences"]
    combined_df = _unique_capped(sequences, np.concatenate(combined), cap, seed)
    residual_df = _unique_capped(sequences, np.concatenate(residual), cap, seed)
    selected, alternate = select_scores(combined_df, residual_df, alpha)
    return selected, alternate, alpha
