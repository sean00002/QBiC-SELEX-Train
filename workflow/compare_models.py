#!/usr/bin/env python3
"""Compare two k-mer models: correlation of their weights and of the variant
effects they predict on a ref/alt sequence table.

  python workflow/compare_models.py retrained.weights.qbic released.weights.qbic \\
      [--sequences example_input_sequences.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qbic_train.kmers import score_sequences  # noqa: E402
from qbic_train.validation import load_weights  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_a")
    p.add_argument("model_b")
    p.add_argument("--sequences", help="CSV with ref_sequence,alt_sequence columns")
    a = p.parse_args(argv)
    wa, wb = load_weights(a.model_a), load_weights(a.model_b)
    if list(wa) != list(wb):
        sys.exit("the two models have different k-mer sets")
    va = np.array(list(wa.values()), dtype=np.float64)
    vb = np.array([wb[k] for k in wa], dtype=np.float64)
    print(f"k-mer weights:      Pearson r = {np.corrcoef(va, vb)[0, 1]:.6f}  "
          f"max |difference| = {np.max(np.abs(va - vb)):.3g}  identical = {bool(np.array_equal(va, vb))}")
    if a.sequences:
        df = pd.read_csv(a.sequences)
        df = df[(df.ref_sequence + df.alt_sequence).str.fullmatch("[ACGT]+")]   # skip variants with N
        ea = score_sequences(wa, df.alt_sequence.values) - score_sequences(wa, df.ref_sequence.values)
        eb = score_sequences(wb, df.alt_sequence.values) - score_sequences(wb, df.ref_sequence.values)
        print(f"predicted effects:  Pearson r = {np.corrcoef(ea, eb)[0, 1]:.6f}  ({len(df)} variants)")


if __name__ == "__main__":
    main()
