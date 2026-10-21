#!/usr/bin/env python3
"""List the cycle pairs to train, as a CSV for ``qbic_selex_train.py --batch``.

  python make_pairs.py --exp_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017 > pairs.csv
  python make_pairs.py --all --per_experiment > experiments.csv   # one row per experiment

Pairs follow the paper's rule: early cycle 0, 1 or 2 (plus the bead-only
selections 0(1)-0(3) for Bhimsaria et al.), late cycle = any later cycle.
"""

import argparse
import sys

import pandas as pd

from qbic_data.manifest import experiment_pairs
from qbic_data.select import read_id_list, select_experiments
from qbic_train.config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp_id", nargs="+", default=[])
    p.add_argument("--tf", nargs="+", default=[], help="TF names (tf or source_tf_label in Table S1)")
    p.add_argument("--source", nargs="+", default=[], help="e.g. jolma2013 yin2017 bhimsaria2023")
    p.add_argument("--model_list", help="file with one model ID per line (e.g. a released collection)")
    p.add_argument("--all", action="store_true", help="every experiment in Table S1")
    p.add_argument("--per_experiment", action="store_true",
                   help="one row per experiment (whole-experiment runs) instead of one row per pair")
    p.add_argument("--config")
    p.add_argument("-o", "--output", help="output CSV (default: stdout)")
    a = p.parse_args(argv)
    manifest = load_config(a.config)["data"]["manifest"]
    model_ids = read_id_list(a.model_list) if a.model_list else []
    if model_ids and not (a.exp_id or a.tf or a.source or a.all):
        from qbic_data.manifest import parse_pair_id
        rows = [dict(zip(["exp_id", "early", "late"], parse_pair_id(m), strict=True)) for m in model_ids]
        df = pd.DataFrame(rows).drop_duplicates()
    else:
        exps = select_experiments(manifest, a.exp_id, a.tf, a.source, model_ids, a.all)
        if a.per_experiment:
            df = pd.DataFrame({"exp_id": exps, "early": "", "late": ""})
        else:
            df = pd.DataFrame([{"exp_id": e, "early": c0, "late": c1}
                               for e in exps for c0, c1 in experiment_pairs(e, manifest)])
    if df.empty:
        sys.exit("nothing selected")
    df.to_csv(a.output or sys.stdout, index=False)
    print(f"{len(df)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
