"""Shared helpers for the model-selection scripts."""

from pathlib import Path

from qbic_data.manifest import parse_pair_id


def score_files(results: Path) -> dict[str, Path]:
    """pair_id -> selected-score CSV in ``<results>/cnn_scores``."""
    return {p.stem: p for p in sorted((Path(results) / "cnn_scores").glob("*.csv")) if ".alternate" not in p.name}


def model_files(results: Path) -> dict[str, Path]:
    """model_id -> ``*.weights.qbic`` in ``<results>/models``."""
    return {p.name.removesuffix(".weights.qbic"): p for p in sorted((Path(results) / "models").glob("*.weights.qbic"))}


def exp_of(pair_or_model_id: str) -> str:
    return parse_pair_id(pair_or_model_id)[0]


def tf_of(model_id: str) -> str:
    """The model-ID prefix (Table S1 ``source_tf_label``)."""
    return model_id.split("_")[0]


def pair_of(model_id: str) -> str:
    exp_id, early, late = parse_pair_id(model_id)
    return f"{exp_id}_{early}_{late}"
