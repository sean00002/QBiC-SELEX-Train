"""One training run: residual CNN for each cycle pair, then 6- and 7-mer OLS models.

Output layout (under ``out_dir``), with ``<pair>`` = ``<exp_id>_<early>_<late>``:

    cnn_scores/<pair>.csv              f_QBiC for each unique read (the OLS input)
    cnn_scores/<pair>.alternate.csv    the other score (only with save_alternate)
    cnn_scores/<pair>.json             alpha, selected score, read counts, settings, versions
    cnn_models/<pair>.pt               residual-model checkpoint (only with save_cnn)
    models/<pair>_<k>mer.weights.qbic  k-mer weights, readable by qbic-selex
    models/<pair>_<k>mer.cov_<n>.npy   their covariance matrix (upper triangle)
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import ols
from .bias_model import load_bias_model, sha256
from .residual import score_reads, train_residual
from .seeding import seed_everything


class TrainingRun:
    """Trains a list of cycle pairs in one random stream, as the original code did.

    The run is seeded once and the bias model created once; the pairs are then
    trained in order with that same bias model. ``read_pair`` returns
    ``(early_reads, late_reads)`` for a pair.
    """

    def __init__(self, cfg, out_dir, device, *, stages=("cnn", "ols"), kmers=(6, 7), save_alternate=False,
                 save_cnn=False, write_cov=True, skip_existing=False, use_gpu_ols=None, log=print):
        self.cfg, self.device, self.log = cfg, device, log
        self.out = Path(out_dir)
        self.stages, self.kmers = tuple(stages), tuple(int(k) for k in kmers)
        self.save_alternate, self.save_cnn, self.write_cov = save_alternate, save_cnn, write_cov
        self.skip_existing = skip_existing
        self.use_gpu_ols = ols.gpu_available() if use_gpu_ols is None else use_gpu_ols
        self.seed = cfg["seed"]
        self.bias_model = None

    # -- paths -------------------------------------------------------------
    def scores_path(self, pair_id):
        return self.out / "cnn_scores" / f"{pair_id}.csv"

    def weights_path(self, pair_id, k):
        return self.out / "models" / f"{pair_id}_{k}mer.weights.qbic"

    def done(self, pair_id) -> bool:
        need = [self.weights_path(p, k) for p in [pair_id] for k in self.kmers] if "ols" in self.stages else []
        if "cnn" in self.stages:
            need.append(self.scores_path(pair_id))
        return bool(need) and all(p.exists() for p in need)

    # -- run ---------------------------------------------------------------
    def run(self, pairs, read_pair, mode: str):
        """Train ``pairs`` (a list of pair IDs) in order. Returns a list of (pair_id, error)."""
        if "cnn" in self.stages:
            seed_everything(self.seed)
            bias_cfg = self.cfg["bias_model"]
            self.bias_model, _ = load_bias_model(bias_cfg["path"], self.device, bias_cfg.get("sha256"))
        failures = []
        for pair_id in pairs:
            if self.skip_existing and self.done(pair_id):
                self.log(f"[skip] {pair_id} (outputs exist)")
                continue
            try:
                if "cnn" in self.stages:
                    self.log(f"[cnn] {pair_id}")
                    early, late = read_pair(pair_id)
                    self.train_cnn(pair_id, early, late, mode)
                if "ols" in self.stages:
                    self.train_ols(pair_id)
            except Exception as e:  # one failed pair must not stop the others
                self.log(f"[failed] {pair_id}: {e}")
                failures.append((pair_id, repr(e)))
        if failures:
            self._log_failures(failures)
        return failures

    def train_cnn(self, pair_id, early, late, mode):
        rcfg = self.cfg["residual"]
        model, dataset, sequences, loss, auroc = train_residual(self.bias_model, early, late, rcfg,
                                                                self.device, self.log)
        selected, alternate, alpha = score_reads(model, dataset, sequences, rcfg, self.device, self.seed)
        out = self.out / "cnn_scores"
        out.mkdir(parents=True, exist_ok=True)
        selected.to_csv(out / f"{pair_id}.csv", index=False)
        if self.save_alternate:
            alternate.to_csv(out / f"{pair_id}.alternate.csv", index=False)
        meta = {
            "pair_id": pair_id, "alpha": alpha,
            "selected_score": "f_RES" if alpha >= 0 else "f_SELEX",
            "alternate_score": "f_SELEX" if alpha >= 0 else "f_RES",
            "selection_rule": "f_RES if alpha >= 0 else f_SELEX",
            "n_early_reads": len(early), "n_late_reads": len(late), "n_scored_unique": int(selected.shape[0]),
            "final_train_loss": loss, "final_train_auroc": auroc,
            "seed": self.seed, "run_mode": mode, "device": str(self.device),
            "residual_settings": self.cfg["residual"],
            "bias_model": Path(self.cfg["bias_model"]["path"]).name,
            "bias_model_sha256": sha256(self.cfg["bias_model"]["path"]),
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
        }
        (out / f"{pair_id}.json").write_text(json.dumps(meta, indent=2))
        if self.save_cnn:
            (self.out / "cnn_models").mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "alpha": alpha,
                        "model_config": {k: self.cfg["residual"][k]
                                         for k in ("num_filters", "kernel_size", "batch_size")},
                        "bias_model_config": load_bias_model_config(self.cfg), "metadata": meta},
                       self.out / "cnn_models" / f"{pair_id}.pt")
        self.log(f"    alpha {alpha:+.4f} -> {meta['selected_score']}; {selected.shape[0]} unique reads scored")

    def train_ols(self, pair_id):
        scores = pd.read_csv(self.scores_path(pair_id))   # read back from disk, as the original did
        models_dir = self.out / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for k in self.kmers:
            data = ols.sample_for_ols(scores, k, self.cfg["ols"]["max_sequences"], self.seed)
            weights, cov = ols.train_kmer_model(data, k, self.use_gpu_ols)
            ols.save_weights(self.weights_path(pair_id, k), weights)
            if self.write_cov:
                ols.save_covariance(models_dir / f"{pair_id}_{k}mer", cov)
            self.log(f"[ols] {pair_id}_{k}mer ({data.shape[0]} reads, {'GPU' if self.use_gpu_ols else 'CPU'})")

    def _log_failures(self, failures):
        path = self.out / "failed_pairs.csv"
        new = pd.DataFrame(failures, columns=["pair_id", "error"])
        if path.exists():
            new = pd.concat([pd.read_csv(path), new])
        self.out.mkdir(parents=True, exist_ok=True)
        new.to_csv(path, index=False)


def load_bias_model_config(cfg) -> dict:
    ckpt = torch.load(cfg["bias_model"]["path"], map_location="cpu", weights_only=False)
    return dict(ckpt["model_config"])
