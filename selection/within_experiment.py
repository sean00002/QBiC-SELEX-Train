"""Within-experiment validation (Methods, step 2).

Each OLS model predicts the CNN scores of every cycle pair of its experiment
(its own pair included), using a random sample of up to 500,000 reads per pair;
the Pearson correlations are written to <results>/validation/within/<model>.csv.

  python -m selection.within_experiment --results results --exp_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017
  python -m selection.within_experiment --results results --model <model_id>   # one Slurm task
"""

import argparse
from pathlib import Path

from qbic_train.config import load_config
from qbic_train.validation import validate

from .common import exp_of, model_files, score_files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", help="output directory of qbic_selex_train.py")
    p.add_argument("--exp_id", nargs="+", default=[], help="validate the models of these experiments (default: all)")
    p.add_argument("--model", nargs="+", default=[], help="validate only these model IDs")
    p.add_argument("--config")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    results = Path(a.results)
    models, scores = model_files(results), score_files(results)
    chosen = [m for m in models if (not a.model or m in a.model) and (not a.exp_id or exp_of(m) in a.exp_id)]
    out = results / "validation" / "within"
    out.mkdir(parents=True, exist_ok=True)
    for model_id in chosen:
        targets = [f for pid, f in scores.items() if exp_of(pid) == exp_of(model_id)]
        table = validate(models[model_id], targets, cfg["validation"]["max_sequences"], cfg["seed"], "within_exp")
        table.to_csv(out / f"{model_id}.csv", index=False)
        print(f"{model_id}: mean r = {table.pearson_r.mean():.4f} over {len(table)} pairs")


if __name__ == "__main__":
    main()
