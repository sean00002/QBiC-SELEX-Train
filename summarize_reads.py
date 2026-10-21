#!/usr/bin/env python3
"""Read statistics per cycle file, and the eligibility rule of the paper.

  python summarize_reads.py --exp_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017 -o read_summary.csv

For each cycle file: reads (all records), unique reads and read length, as in the
original summary script. An experiment is flagged eligible if at least two of its
cycles (counting the initial library) have more than 10,000 reads; Methods:
"4,362 experiments with at least two available cycles and more than 10,000 reads
per cycle".
"""

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd

from qbic_data import manifest as mf
from qbic_data.select import select_experiments
from qbic_train.config import load_config

MIN_READS = 10_000


def file_stats(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        if path.name.endswith(".txt"):           # one read per line (rm2017)
            seqs = [line.strip() for line in f]
        else:                                     # FASTQ: the second line of every four
            seqs = [line.strip() for i, line in enumerate(f) if i % 4 == 1]
    return {"reads": len(seqs), "unique_reads": len(set(seqs)), "read_length": len(seqs[0]) if seqs else 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp_id", nargs="+", default=[])
    p.add_argument("--source", nargs="+", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--reads_dir")
    p.add_argument("--config")
    p.add_argument("-o", "--output", default="read_summary.csv")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    manifest, reads_dir = cfg["data"]["manifest"], a.reads_dir or cfg["data"]["reads_dir"]
    rows = []
    for exp_id in select_experiments(manifest, a.exp_id, (), a.source, (), a.all):
        for f in mf.experiment_files(exp_id, manifest):
            path = f.path(reads_dir)
            stats = file_stats(path) if path.exists() else {"reads": None, "unique_reads": None, "read_length": None}
            rows.append({"exp_id": exp_id, "cycle": f.cycle, "source": f.source, "filename": f.filename, **stats})
    if not rows:
        sys.exit("nothing selected")
    df = pd.DataFrame(rows)
    ok = df.assign(ok=df.reads.fillna(0) > MIN_READS).groupby("exp_id").ok.sum() >= 2
    df["experiment_eligible"] = df.exp_id.map(ok)
    df.to_csv(a.output, index=False)
    n_missing = df.reads.isna().sum()
    print(f"{df.exp_id.nunique()} experiments, {int(ok.sum())} eligible; "
          f"{n_missing} files not downloaded -> {a.output}")


if __name__ == "__main__":
    main()
