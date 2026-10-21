#!/usr/bin/env python3
"""Train a universal bias-model candidate on one TF-free paired-cycle dataset
(Supplementary Methods: "Bias-model parameter optimization and selection").

  python qbic_train_bias_model.py --exp_id beadOnlyControl_NA_TG40NCATATG_AATBA_jolma2025 --early 0 --late 2

Runs the 108-configuration grid search with 3-fold cross-validation, then
trains the best configuration on all reads. Writes
<out_dir>/<pair>_hyperparameter_results.csv and <out_dir>/<pair>_CNN<k>K<b>B<f>F_L<l>_D<d>.pt.
Compare candidates with ``python -m selection.bias_model_matrix``.

The paper's final bias model is bias_model/ in this repository; this script
documents how candidates were made and is not needed to train TF models.
Note: the original grid search did not fix PyTorch's seed; this script seeds
with the config seed, so its candidates are reproducible but not bit-identical
to the paper's.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qbic_data import manifest as mf
from qbic_train import bias_model as bm
from qbic_train.config import load_config
from qbic_train.reads import read_sequences
from qbic_train.seeding import pick_device, seed_everything


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp_id", help="TF-free experiment ID from Table S1")
    p.add_argument("--early", "-c", "--control_round", dest="early")
    p.add_argument("--late", "-e", "--enriched_round", dest="late")
    p.add_argument("--early_reads", help="or: your own early-cycle reads")
    p.add_argument("--late_reads", help="or: your own late-cycle reads")
    p.add_argument("--name", help="output name for --early_reads/--late_reads")
    p.add_argument("--out_dir", "-o", default="bias_candidates")
    p.add_argument("--data_dir")
    p.add_argument("--config")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    bcfg = cfg["bias_training"]
    device = pick_device(a.device)
    seed_everything(cfg["seed"])

    if a.early_reads:
        early, late = read_sequences(a.early_reads), read_sequences(a.late_reads)
        pair_id = a.name or Path(a.late_reads).name.split(".")[0]
    else:
        reads_dir, manifest = a.data_dir or cfg["data"]["reads_dir"], cfg["data"]["manifest"]
        early = mf.load_cycle_reads(a.exp_id, a.early, reads_dir, manifest)
        late = mf.load_cycle_reads(a.exp_id, a.late, reads_dir, manifest)
        pair_id = f"{a.exp_id}_{a.early}_{a.late}"
    X = np.array(early + late)
    y = np.concatenate([np.zeros(len(early), dtype=np.float32), np.ones(len(late), dtype=np.float32)])

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = bm.grid(bcfg)
    results = []
    for i in range(len(grid)):
        params = grid[i]
        try:
            results.append({"config": i, **bm.cross_validate_config(params, X, y, bcfg, device)})
            print(f"config {i}: {params} -> mean CV AUROC {results[-1]['avg_auroc']:.4f}")
        except RuntimeError as e:   # e.g. out of GPU memory for the largest settings
            print(f"config {i}: {params} failed: {e}")
            results.append({"config": i, **params, "avg_auroc": np.nan, "std_auroc": np.nan, "error": str(e)})
    table = pd.DataFrame(results)
    table.to_csv(out / f"{pair_id}_hyperparameter_results.csv", index=False)

    best = table.loc[table["avg_auroc"].idxmax()]
    params = {k: (bool(best[k]) if k == "use_dropout" else int(best[k]))
              for k in ("batch_size", "num_filters", "kernel_size", "n_conv_layers", "use_dropout")}
    print(f"best configuration: {params} (mean CV AUROC {best['avg_auroc']:.4f})")
    model, loss, auroc = bm.train_final(params, X, y, bcfg, device)
    path = out / bm.checkpoint_name(pair_id, params)
    torch.save({"model_state_dict": model.state_dict(), "model_config": params, "final_loss": loss,
                "final_auroc": auroc, "best_val_auroc": float(best["avg_auroc"])}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
