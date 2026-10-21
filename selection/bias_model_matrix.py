"""Compare universal bias-model candidates (Supplementary Fig. 2).

Each candidate classifies the early- vs late-cycle reads (unique reads, N
removed) of every TF-free cycle pair; the AUROC matrix and each candidate's
mean AUROC are written, and the best candidate is reported. The paper's
choice is the model in bias_model/.

  python -m selection.bias_model_matrix --models bias_candidates/*.pt \\
      --pairs beadOnlyControl_NA_TG40NCATATG_AATBA_jolma2025_0_1 beadOnlyControl_NA_TG40NCATATG_AATBA_jolma2025_0_2 ...
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qbic_data import manifest as mf
from qbic_train.bias_model import evaluate_auroc, load_bias_model
from qbic_train.config import load_config
from qbic_train.reads import read_sequences
from qbic_train.seeding import pick_device


def unique_pair_reads(pair_id, reads_dir, manifest):
    """Unique reads of both cycles (Bhimsaria reads cut to 20 bp after de-duplication)."""
    exp_id, early, late = mf.parse_pair_id(pair_id)
    out = []
    for cycle in (early, late):
        f = mf.cycle_file(exp_id, cycle, manifest)
        reads = list(set(read_sequences(f.path(reads_dir))))
        if f.source == mf.BHIMSARIA:
            reads = [r[:mf.BHIMSARIA_TRIM] for r in reads]
        out.append(reads)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="candidate checkpoints")
    p.add_argument("--pairs", nargs="+", required=True, help="TF-free pair IDs <exp_id>_<early>_<late>")
    p.add_argument("--out", default="bias_candidates/bias_model_matrix.csv")
    p.add_argument("--data_dir")
    p.add_argument("--config")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    reads_dir, manifest = a.data_dir or cfg["data"]["reads_dir"], cfg["data"]["manifest"]
    device = pick_device(a.device)
    data = {pid: unique_pair_reads(pid, reads_dir, manifest) for pid in a.pairs}
    rows = []
    for path in a.models:
        model, ckpt = load_bias_model(path, device)
        for pid, (early, late) in data.items():
            labels = np.concatenate([np.zeros(len(early), dtype=np.float32), np.ones(len(late), dtype=np.float32)])
            auroc = evaluate_auroc(model, np.array(early + late), labels, ckpt["model_config"]["batch_size"], device)
            rows.append({"model": Path(path).name, "pair_id": pid, "auroc": auroc})
            print(f"{Path(path).name} on {pid}: AUROC {auroc:.4f}")
    matrix = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(a.out, index=False)
    means = matrix.groupby("model")["auroc"].mean().sort_values(ascending=False)
    print("\nmean AUROC per candidate:\n" + means.to_string())
    print(f"\nbest candidate: {means.index[0]}")


if __name__ == "__main__":
    main()
