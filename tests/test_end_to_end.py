"""Tiny CPU training runs with the bundled bias model: outputs, determinism, seeds."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from qbic_train.bias_model import load_bias_model, sha256
from qbic_train.config import load_config
from qbic_train.pipeline import TrainingRun

MOTIF = "CCGGAAGT"


def _reads(n, seed, motif=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = "".join(rng.choice(list("ACGT"), 20))
        if motif:
            i = rng.integers(0, 12)
            s = s[:i] + MOTIF + s[i + 8:]
        out.append(s)
    return out


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["residual"]["num_epochs"] = 1
    return c


PAIRS = {"TOY_x_BC_B_src_0_1": (_reads(300, 1), _reads(300, 2, motif=True)),
         "TOY_x_BC_B_src_0_2": (_reads(300, 1), _reads(300, 3, motif=True))}


def _run(cfg, out, pairs=("TOY_x_BC_B_src_0_1",), **kw):
    run = TrainingRun(cfg, out, torch.device("cpu"), kmers=(3,), use_gpu_ols=False, **kw)
    return run.run(list(pairs), PAIRS.__getitem__, "experiment" if len(pairs) > 1 else "pair")


def test_bundled_bias_model_matches_the_paper(cfg):
    model, ckpt = load_bias_model(cfg["bias_model"]["path"], "cpu", cfg["bias_model"]["sha256"])
    assert sha256(cfg["bias_model"]["path"]) == cfg["bias_model"]["sha256"]
    assert ckpt["model_config"] == {"num_filters": 512, "kernel_size": 11, "batch_size": 128,
                                    "n_conv_layers": 2, "use_dropout": True}
    assert not any(p.requires_grad for p in model.parameters())


def test_outputs(cfg, tmp_path):
    assert _run(cfg, tmp_path, save_alternate=True, save_cnn=True) == []
    pid = "TOY_x_BC_B_src_0_1"
    scores = pd.read_csv(tmp_path / "cnn_scores" / f"{pid}.csv")
    assert list(scores.columns) == ["sequence", "score"] and scores.sequence.is_unique
    meta = json.loads((tmp_path / "cnn_scores" / f"{pid}.json").read_text())
    assert meta["selected_score"] == ("f_RES" if meta["alpha"] >= 0 else "f_SELEX")
    assert meta["n_early_reads"] == 300 and meta["run_mode"] == "pair" and meta["seed"] == 42
    assert (tmp_path / "cnn_scores" / f"{pid}.alternate.csv").exists()
    assert (tmp_path / "cnn_models" / f"{pid}.pt").exists()
    with open(tmp_path / "models" / f"{pid}_3mer.weights.qbic", "rb") as f:
        assert len(pickle.load(f)) == 32
    assert (tmp_path / "models" / f"{pid}_3mer.cov_32.npy").exists()


def _outputs(d, pid="TOY_x_BC_B_src_0_1"):
    with open(d / "models" / f"{pid}_3mer.weights.qbic", "rb") as f:
        return (d / "cnn_scores" / f"{pid}.csv").read_bytes(), pickle.load(f)


def test_same_seed_same_files_and_seed_is_used(cfg, tmp_path):
    _run(cfg, tmp_path / "a")
    _run(cfg, tmp_path / "b")
    assert _outputs(tmp_path / "a") == _outputs(tmp_path / "b")
    other = dict(cfg, seed=7)
    _run(other, tmp_path / "c")
    assert _outputs(tmp_path / "a")[0] != _outputs(tmp_path / "c")[0]


def test_experiment_runs_share_one_random_stream(cfg, tmp_path):
    _run(cfg, tmp_path / "exp", pairs=("TOY_x_BC_B_src_0_1", "TOY_x_BC_B_src_0_2"))
    _run(cfg, tmp_path / "single", pairs=("TOY_x_BC_B_src_0_2",))
    pid = "TOY_x_BC_B_src_0_2"
    # the second pair of a whole-experiment run differs from the same pair trained on its own
    assert _outputs(tmp_path / "exp", pid)[0] != _outputs(tmp_path / "single", pid)[0]
    # the first pair is the same as a single-pair run of it
    _run(cfg, tmp_path / "first", pairs=("TOY_x_BC_B_src_0_1",))
    assert _outputs(tmp_path / "exp")[0] == _outputs(tmp_path / "first")[0]


def test_failed_pair_is_logged_and_others_continue(cfg, tmp_path):
    def read_pair(pid):
        if pid.endswith("_0_2"):
            raise FileNotFoundError("reads missing")
        return PAIRS[pid]
    run = TrainingRun(cfg, tmp_path, torch.device("cpu"), kmers=(3,), use_gpu_ols=False)
    failures = run.run(["TOY_x_BC_B_src_0_2", "TOY_x_BC_B_src_0_1"], read_pair, "pair")
    assert [f[0] for f in failures] == ["TOY_x_BC_B_src_0_2"]
    assert (tmp_path / "models" / "TOY_x_BC_B_src_0_1_3mer.weights.qbic").exists()
    assert pd.read_csv(tmp_path / "failed_pairs.csv").pair_id.tolist() == ["TOY_x_BC_B_src_0_2"]
