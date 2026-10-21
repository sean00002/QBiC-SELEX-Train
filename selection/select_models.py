"""Model selection from the validation results (Methods; Tables S2 and S3).

  python -m selection.select_models within --results results   # top 2 models per experiment (Table S2)
  python -m selection.select_models cross  --results results   # best model per TF (Table S3)

``within`` averages each model's within-experiment correlations and keeps the
two best models of each experiment. ``cross`` averages the cross-experiment
correlations of those models and keeps the best one per TF. Ties keep the
model that sorts first, as in the original analysis.
"""

import argparse
from pathlib import Path

import pandas as pd

from qbic_data.manifest import load_manifest
from qbic_train.config import load_config

from .common import exp_of, tf_of


def _read_dir(directory: Path) -> pd.DataFrame:
    files = sorted(Path(directory).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no validation results in {directory}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def top2_per_experiment(within: pd.DataFrame) -> pd.DataFrame:
    df = within[["model_id", "target_pair_id", "pearson_r"]].drop_duplicates()
    mean = df.groupby("model_id", as_index=False)["pearson_r"].mean()
    mean["exp_id"] = mean.model_id.map(exp_of)
    mean = mean.sort_values(["exp_id", "pearson_r"], ascending=[True, False], kind="stable")
    top = mean.groupby("exp_id", sort=True).head(2)
    return pd.DataFrame({"model_id": top.model_id.values, "source_tf_label": top.model_id.map(tf_of).values,
                         "within_exp_pearsonr": top.pearson_r.values})


def best_per_tf(cross: pd.DataFrame, top2: pd.DataFrame, manifest=None) -> pd.DataFrame:
    mean = cross.groupby("model_id", as_index=False)["pearson_r"].mean()
    mean["source_tf_label"] = mean.model_id.map(tf_of)
    best = mean.loc[mean.groupby("source_tf_label")["pearson_r"].idxmax()].reset_index(drop=True)
    best = best.rename(columns={"pearson_r": "cross_exp_pearsonr"})
    best = best.merge(top2[["model_id", "within_exp_pearsonr"]], on="model_id", how="left")
    s1 = load_manifest(manifest).drop_duplicates("exp_id").set_index("exp_id")
    exp = best.model_id.map(exp_of)
    best["hgnc_symbol"] = exp.map(s1["hgnc_symbol"]).values
    best["ensembl_gene_id"] = exp.map(s1["ensembl_gene_id"]).values
    return best[["model_id", "source_tf_label", "hgnc_symbol", "ensembl_gene_id",
                 "cross_exp_pearsonr", "within_exp_pearsonr"]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["within", "cross"])
    p.add_argument("--results", default="results")
    p.add_argument("--config")
    a = p.parse_args(argv)
    results = Path(a.results)
    out = results / "selection"
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "within":
        table = top2_per_experiment(_read_dir(results / "validation" / "within"))
        path = out / "within_experiment_top2.csv"
    else:
        top2 = pd.read_csv(out / "within_experiment_top2.csv")
        manifest = load_config(a.config)["data"]["manifest"]
        table = best_per_tf(_read_dir(results / "validation" / "cross"), top2, manifest)
        path = out / "primary_models.csv"
    table.to_csv(path, index=False)
    print(f"{len(table)} models -> {path}")


if __name__ == "__main__":
    main()
