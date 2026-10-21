"""Universal bias model: loading the trained model, and training new candidates
(Methods: "Universal bias model architecture and training"; Supplementary
Methods: "Bias-model parameter optimization and selection")."""

import hashlib
import warnings
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .networks import SequenceDataset, UpdatedCNN


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bias_model(path, device, expected_sha256: str | None = None):
    """Load a bias-model checkpoint as a frozen ``UpdatedCNN``.

    The network is created before its weights are loaded, as in the original
    code; this matters because creating it draws random numbers, and later
    random draws (residual-model initialisation, shuffling) follow it.
    Returns ``(model, checkpoint)``.
    """
    if expected_sha256 and sha256(path) != expected_sha256:
        warnings.warn(f"{path} is not the universal bias model used in the paper (sha256 differs)", stacklevel=2)
    # Trusted local checkpoint; it stores numpy scalars, which weights_only=True rejects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["model_config"]
    model = UpdatedCNN(num_filters=cfg["num_filters"], kernel_size=cfg["kernel_size"],
                       n_conv_layers=cfg["n_conv_layers"], use_dropout=cfg["use_dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, checkpoint


def evaluate_auroc(model, sequences, labels, batch_size, device) -> float:
    """AUROC of a model's sigmoid output for early (0) vs late (1) reads."""
    loader = DataLoader(SequenceDataset(sequences, labels), batch_size=batch_size)
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds.append(torch.sigmoid(model(X_batch.to(device)).squeeze(-1)).cpu().numpy())
            ys.append(y_batch.numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(preds))


def _class_weighted_loss(loader, device):
    y_all = torch.cat([y for _, y in loader])
    pos_weight = torch.tensor([(1 - y_all.mean()) / y_all.mean()]).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _train_epoch(model, loader, criterion, optimizer, device, clip):
    model.train()
    total_loss, preds, ys = 0.0, [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        with torch.no_grad():
            preds.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(y_batch.cpu().numpy())
    return total_loss / len(loader.dataset), roc_auc_score(np.concatenate(ys), np.concatenate(preds))


def _make_optimizer(model, cfg):
    return optim.Adam(model.parameters(), lr=cfg["learning_rate"], eps=cfg["adam_eps"], amsgrad=cfg["amsgrad"])


def cross_validate_config(params, X, y, cfg, device) -> dict:
    """k-fold CV of one configuration, with early stopping on validation AUROC."""
    kf = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["cv_random_state"])
    fold_aurocs = []
    for train_idx, val_idx in kf.split(X):
        train_loader = DataLoader(SequenceDataset(X[train_idx], y[train_idx]),
                                  batch_size=params["batch_size"], shuffle=True)
        model = UpdatedCNN(num_filters=params["num_filters"], kernel_size=params["kernel_size"],
                           n_conv_layers=params["n_conv_layers"], use_dropout=params["use_dropout"]).to(device)
        criterion = _class_weighted_loss(train_loader, device)
        optimizer = _make_optimizer(model, cfg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=cfg["lr_factor"],
                                                         patience=cfg["lr_patience"])
        best, waited = 0.0, 0
        for _ in range(cfg["cv_epochs"]):
            _train_epoch(model, train_loader, criterion, optimizer, device, cfg["grad_clip_norm"])
            val_auroc = evaluate_auroc(model, X[val_idx], y[val_idx], params["batch_size"], device)
            scheduler.step(val_auroc)
            if val_auroc > best:
                best, waited = val_auroc, 0
            else:
                waited += 1
            if waited >= cfg["early_stopping_patience"]:
                break
        fold_aurocs.append(float(best))
    return {**params, "avg_auroc": float(np.mean(fold_aurocs)), "std_auroc": float(np.std(fold_aurocs))}


def grid(cfg) -> list[dict]:
    g = cfg["grid"]
    keys = ["batch_size", "num_filters", "kernel_size", "n_conv_layers", "use_dropout"]
    return [dict(zip(keys, values, strict=True)) for values in product(*(g[k] for k in keys))]


def train_final(params, X, y, cfg, device):
    """Train the chosen configuration on all reads; returns (model, final_loss, final_auroc)."""
    loader = DataLoader(SequenceDataset(X, y), batch_size=params["batch_size"], shuffle=True)
    model = UpdatedCNN(num_filters=params["num_filters"], kernel_size=params["kernel_size"],
                       n_conv_layers=params["n_conv_layers"], use_dropout=params["use_dropout"]).to(device)
    criterion = _class_weighted_loss(loader, device)
    optimizer = _make_optimizer(model, cfg)
    for epoch in range(1, cfg["final_epochs"] + 1):
        loss, auroc = _train_epoch(model, loader, criterion, optimizer, device, cfg["grad_clip_norm"])
        print(f"Epoch {epoch}/{cfg['final_epochs']}, Loss: {loss:.4f}, AUROC: {auroc:.4f}")
    return model, loss, auroc


def checkpoint_name(pair_id: str, params: dict) -> str:
    """File name used for bias-model candidates, e.g. ..._0_2_CNN11K128B512F_L2_DTrue.pt"""
    return (f"{pair_id}_CNN{params['kernel_size']}K{params['batch_size']}B{params['num_filters']}F"
            f"_L{params['n_conv_layers']}_D{params['use_dropout']}.pt")
