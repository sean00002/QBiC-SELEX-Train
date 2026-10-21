"""Cross-experiment validation (Methods, step 3).

Each top-2 model (from ``select_models within``) predicts the CNN scores of the
top-2 cycle pairs of every experiment of the same TF, its own experiment
included; Pearson correlations go to <results>/validation/cross/<model>.csv.

  python -m selection.cross_experiment --results results
  python -m selection.cross_experiment --results results --model <model_id>   # one Slurm task
"""

import argparse
from pathlib import Path

import pandas as pd

from qbic_train.config import load_config
from qbic_train.validation import validate

from .common import model_files, pair_of, score_files, tf_of


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--top2", help="default: <results>/selection/within_experiment_top2.csv")
    p.add_argument("--model", nargs="+", default=[], help="validate only these model IDs")
    p.add_argument("--config")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    results = Path(a.results)
    top2 = pd.read_csv(a.top2 or results / "selection" / "within_experiment_top2.csv")
    models, scores = model_files(results), score_files(results)
    out = results / "validation" / "cross"
    out.mkdir(parents=True, exist_ok=True)
    for model_id in top2.model_id:
        if a.model and model_id not in a.model:
            continue
        tf = tf_of(model_id)
        target_pairs = sorted({pair_of(m) for m in top2.model_id if tf_of(m) == tf})
        missing = [pid for pid in target_pairs if pid not in scores]
        if missing:
            raise FileNotFoundError(f"CNN score files missing for {missing[:3]}")
        table = validate(models[model_id], [scores[pid] for pid in target_pairs],
                         cfg["validation"]["max_sequences"], cfg["seed"], "cross_exp")
        table.to_csv(out / f"{model_id}.csv", index=False)
        print(f"{model_id}: mean r = {table.pearson_r.mean():.4f} over {len(table)} pairs")


if __name__ == "__main__":
    main()
