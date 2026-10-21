#!/usr/bin/env python3
"""Train QBiC-SELEX models: a residual CNN for each cycle pair, then 6- and 7-mer OLS models.

Paper data (cycle files looked up in Table S1):
  python qbic_selex_train.py --exp_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017            # each pair of the experiment
  python qbic_selex_train.py --exp_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017 --early 0 --late 4
  python qbic_selex_train.py --model_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer  # the pair behind a model
  python qbic_selex_train.py --batch pairs.csv --row 7     # one row of make_pairs.py output

Your own reads (FASTQ(.gz), FASTA, .txt or .csv):
  python qbic_selex_train.py --early_reads cycle0.fastq.gz --late_reads cycle4.fastq.gz --name MYTF_0_4
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from qbic_train.config import load_config
from qbic_train.pipeline import TrainingRun
from qbic_train.reads import read_sequences
from qbic_train.seeding import pick_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("what to train (paper data)")
    src.add_argument("--exp_id", help="experiment ID from Table S1")
    src.add_argument("--early", "--control_round", dest="early",
                     help="early cycle (0, 1, 2, or 0(1)/0(2)/0(3) for Bhimsaria)")
    src.add_argument("--late", "--enriched_round", dest="late", help="late cycle")
    src.add_argument("--pair_id", help="<exp_id>_<early>_<late>")
    src.add_argument("--model_id", help="<exp_id>_<early>_<late>_<k>mer: train that model's pair")
    src.add_argument("--batch", help="CSV with exp_id[,early,late] rows (see make_pairs.py)")
    src.add_argument("--row", type=int, help="with --batch: run only this 1-based row (e.g. $SLURM_ARRAY_TASK_ID)")
    src.add_argument("--download", action="store_true", help="download missing paper reads first")
    own = p.add_argument_group("what to train (your own reads)")
    own.add_argument("--early_reads", "--negative_seqs", dest="early_reads", help="early-cycle reads")
    own.add_argument("--late_reads", "--positive_seqs", dest="late_reads", help="late-cycle reads")
    own.add_argument("--name", help="output name for --early_reads/--late_reads (default: <late file stem>)")
    own.add_argument("--trim", type=int, help="cut reads to their first N bases")
    run = p.add_argument_group("run options")
    run.add_argument("--stage", choices=["cnn", "ols", "all", "residual"], default="all",
                     help="cnn = residual CNN + scores; ols = k-mer models from existing scores; all = both")
    run.add_argument("--kmer", type=int, nargs="+", choices=[6, 7, 8], help="k-mer sizes (default from config: 6 7)")
    run.add_argument("--out_dir", default="results", help="output directory (default: results)")
    run.add_argument("--data_dir", help="paper reads directory (default from config: data/reads)")
    run.add_argument("--config", help="configuration file (default: config/config.yaml)")
    run.add_argument("--bias_model", help="use a different bias-model checkpoint")
    run.add_argument("--seed", type=int, help="override the seed in the config")
    run.add_argument("--device", default="auto", help="CNN device: auto, cpu, cuda, cuda:1, ...")
    run.add_argument("--ols_device", choices=["auto", "gpu", "cpu"], default="auto",
                     help="OLS on GPU (CuPy) or CPU (NumPy); auto = GPU if CuPy works")
    run.add_argument("--save_alternate", "--save_uncorrected", dest="save_alternate", action="store_true",
                     help="also save the alternate CNN score (the other of f_RES / f_SELEX)")
    run.add_argument("--save_cnn", "--save_models", dest="save_cnn", action="store_true",
                     help="save residual-model checkpoints")
    run.add_argument("--no_cov", action="store_true", help="do not write covariance matrices")
    run.add_argument("--skip_existing", action="store_true", help="skip pairs whose outputs already exist")
    return p.parse_args(argv)


def build_jobs(args, cfg):
    """Return ``(pairs, read_pair, needed)``: the pair IDs to train (each in its own fresh run),
    a function giving a pair's early and late reads, and the (exp_id, early, late) triples."""
    from qbic_data import manifest as mf

    manifest = cfg["data"]["manifest"]
    reads_dir = args.data_dir or cfg["data"]["reads_dir"]

    def read_paper_pair(pair_id):
        exp_id, early, late = mf.parse_pair_id(pair_id)
        return (mf.load_cycle_reads(exp_id, early, reads_dir, manifest),
                mf.load_cycle_reads(exp_id, late, reads_dir, manifest))

    if args.early_reads or args.late_reads:
        if not (args.early_reads and args.late_reads):
            sys.exit("--early_reads and --late_reads go together")
        name = args.name or Path(args.late_reads).name.split(".")[0]
        reads = {name: (read_sequences(args.early_reads, args.trim), read_sequences(args.late_reads, args.trim))}
        return [name], reads.__getitem__, []

    targets = []   # (exp_id, early or None, late or None)
    if args.batch:
        df = pd.read_csv(args.batch, dtype=str, keep_default_na=False)
        df = df.rename(columns={"control_round": "early", "enriched_round": "late"})
        rows = [df.iloc[args.row - 1]] if args.row else [r for _, r in df.iterrows()]
        targets = [(r["exp_id"], r.get("early") or None, r.get("late") or None) for r in rows]
    elif args.pair_id or args.model_id:
        targets = [(*mf.parse_pair_id(args.pair_id or args.model_id),)]
    elif args.exp_id:
        targets = [(args.exp_id, args.early, args.late)]
    else:
        sys.exit("say what to train: --exp_id, --pair_id, --model_id, --batch, or --early_reads/--late_reads")

    pairs = []
    for exp_id, early, late in targets:
        all_pairs = [f"{exp_id}_{e}_{c}" for e, c in mf.experiment_pairs(exp_id, manifest)]
        if early is None and late is None:
            pairs += all_pairs
        elif early is not None and late is not None:
            pair = f"{exp_id}_{early}_{late}"
            if pair not in all_pairs:
                sys.exit(f"{pair} is not a valid cycle pair of {exp_id} (pairs: {', '.join(all_pairs)})")
            pairs.append(pair)
        else:
            sys.exit("give both --early and --late, or neither (all pairs of the experiment)")
    return pairs, read_paper_pair, [mf.parse_pair_id(pid) for pid in pairs]


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.bias_model:
        cfg["bias_model"] = {"path": args.bias_model, "sha256": None}
    stages = {"cnn": ("cnn",), "residual": ("cnn",), "ols": ("ols",), "all": ("cnn", "ols")}[args.stage]
    pairs, read_pair, needed = build_jobs(args, cfg)

    if args.download and needed and "cnn" in stages:
        from qbic_data import download, manifest as mf
        files = mf.files_for_pairs(needed, cfg["data"]["manifest"])
        failures = download.download_files(files, args.data_dir or cfg["data"]["reads_dir"])
        if failures:
            sys.exit(f"{len(failures)} downloads failed")

    use_gpu_ols = {"auto": None, "gpu": True, "cpu": False}[args.ols_device]
    run = TrainingRun(cfg, args.out_dir, pick_device(args.device), stages=stages,
                      kmers=args.kmer or cfg["ols"]["kmers"], save_alternate=args.save_alternate,
                      save_cnn=args.save_cnn, write_cov=not args.no_cov, skip_existing=args.skip_existing,
                      use_gpu_ols=use_gpu_ols)
    failures = run.run(pairs, read_pair)
    if failures:
        print(f"{len(failures)} pair(s) failed; see {Path(args.out_dir) / 'failed_pairs.csv'}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
