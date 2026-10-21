#!/usr/bin/env python3
"""Download the raw HT-SELEX reads used in the paper (files listed in Table S1).

  python download_reads.py --exp_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017
  python download_reads.py --model_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer   # just that pair's 2 files
  python download_reads.py --model_list primary_997.txt                            # pairs behind a collection
  python download_reads.py --all --dry_run                                         # everything (~310 GB)

Files go to <reads_dir>/<source>/<file name in Table S1>, which is where
qbic_selex_train.py looks for them. Each experiment's cycle-0 library is included.
"""

import argparse
import sys

from qbic_data import download, manifest as mf
from qbic_data.select import read_id_list, select_experiments
from qbic_train.config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp_id", nargs="+", default=[])
    p.add_argument("--tf", nargs="+", default=[])
    p.add_argument("--source", nargs="+", default=[])
    p.add_argument("--model_id", nargs="+", default=[], help="model or pair IDs: download only their two cycles")
    p.add_argument("--model_list", help="file with one model ID per line")
    p.add_argument("--all", action="store_true")
    p.add_argument("--reads_dir", help="default from config: data/reads")
    p.add_argument("--dry_run", action="store_true", help="list the files without downloading")
    p.add_argument("--config")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    manifest, reads_dir = cfg["data"]["manifest"], a.reads_dir or cfg["data"]["reads_dir"]

    model_ids = list(a.model_id) + (read_id_list(a.model_list) if a.model_list else [])
    files = mf.files_for_pairs([mf.parse_pair_id(m) for m in model_ids], manifest)
    for exp_id in select_experiments(manifest, a.exp_id, a.tf, a.source, (), a.all):
        files += mf.experiment_files(exp_id, manifest)
    unique = list({(f.source, f.filename): f for f in files}.values())
    if not unique:
        sys.exit("nothing selected")
    todo = [f for f in unique if not f.path(reads_dir).exists()]
    print(f"{len(unique)} files selected, {len(todo)} to download into {reads_dir}")
    if a.dry_run:
        for f in todo:
            print(f"  {f.source}/{f.filename}\t{f.submitted_ftp}")
        return
    failures = download.download_files(todo, reads_dir)
    if failures:
        sys.exit(f"{len(failures)} downloads failed")


if __name__ == "__main__":
    main()
