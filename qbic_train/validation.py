"""Model validation (Methods: "Paired-cycle design and within-/cross-experiment
validation strategy"): an OLS model predicts the CNN scores of other cycle
pairs, and performance is the Pearson correlation."""

import pickle

import pandas as pd
from scipy import stats

from .kmers import score_sequences


def load_weights(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def model_vs_scores(weights: dict, score_file, max_sequences: int, seed: int) -> float:
    """Pearson r between CNN scores in ``score_file`` and the OLS prediction,
    on a random sample (without replacement) of at most ``max_sequences`` reads."""
    df = pd.read_csv(score_file)
    if df.shape[0] > max_sequences:
        df = df.sample(n=max_sequences, random_state=seed)
    predicted = score_sequences(weights, df["sequence"].values)
    return stats.pearsonr(df["score"].values, predicted)[0]


def validate(model_path, score_files, max_sequences, seed, scope: str) -> pd.DataFrame:
    weights = load_weights(model_path)
    model_id = model_path.name.removesuffix(".weights.qbic")
    rows = [{"model_id": model_id, "target_pair_id": f.name.removesuffix(".csv"),
             "pearson_r": model_vs_scores(weights, f, max_sequences, seed), "scope": scope}
            for f in sorted(score_files)]
    return pd.DataFrame(rows)
