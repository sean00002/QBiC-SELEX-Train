"""k-mer features and the OLS module, including the file formats qbic-selex reads."""

import pickle
from itertools import product

import numpy as np
import pandas as pd
import pytest

from qbic_train import kmers, ols


def test_feature_counts_and_order():
    assert len(kmers.canonical_kmers(6)) == 2080
    assert len(kmers.canonical_kmers(7)) == 8192
    for k in (3, 6):
        feats = kmers.canonical_kmers(k)
        assert list(feats) == sorted(feats)
        assert all(f >= kmers.reverse_complement(f) for f in feats)   # larger of each reverse-complement pair


def test_count_matrix_matches_naive_counting():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 15)) for _ in range(30)]
    X = kmers.kmer_count_matrix(seqs, 4)
    assert X.dtype == np.float32
    feats = kmers.canonical_kmers(4)
    for row, s in zip(X, seqs, strict=True):
        naive = dict.fromkeys(feats, 0)
        for km in kmers.sliding_window(s, 4):
            naive[kmers.canonical(km)] += 1
        assert row.tolist() == [naive[f] for f in feats]


def test_score_sequences_sums_weights_of_both_orientations():
    w = {f: np.float32(i) for i, f in enumerate(kmers.canonical_kmers(2))}
    seq = "ACGT"
    expected = sum(float(w[kmers.canonical(km)]) for km in kmers.sliding_window(seq, 2))
    assert kmers.score_sequences(w, [seq])[0] == pytest.approx(expected)


def test_ols_recovers_planted_coefficients_and_writes_qbic_selex_files(tmp_path):
    rng = np.random.default_rng(1)
    k = 3
    seqs = ["".join(rng.choice(list("ACGT"), 20)) for _ in range(3000)]
    beta_true = rng.normal(size=len(kmers.canonical_kmers(k)))
    y = kmers.kmer_count_matrix(seqs, k) @ beta_true + rng.normal(scale=0.01, size=len(seqs))
    weights, cov = ols.train_kmer_model(pd.DataFrame({"sequence": seqs, "score": y}), k, use_gpu=False)
    assert list(weights) == list(kmers.canonical_kmers(k))
    assert all(isinstance(v, np.float32) for v in weights.values())
    np.testing.assert_allclose([weights[f] for f in kmers.canonical_kmers(k)], beta_true, atol=1e-2)

    ols.save_weights(tmp_path / "m.weights.qbic", weights)
    path = ols.save_covariance(tmp_path / "m", cov)
    n = len(weights)
    assert path.endswith(f"m.cov_{n}.npy")
    upper = np.load(path)
    assert upper.dtype == np.float32 and upper.shape == (n * (n + 1) // 2,)
    with open(tmp_path / "m.weights.qbic", "rb") as f:
        assert pickle.load(f) == weights


def test_files_load_in_qbic_selex(tmp_path):
    qmodels = pytest.importorskip("qbic_selex.models")
    k = 3
    weights = {f: np.float32(i / 10) for i, f in enumerate(kmers.canonical_kmers(k))}
    ols.save_weights(tmp_path / "TF_x_BC_B_src_0_1_3mer.weights.qbic", weights)
    ols.save_covariance(tmp_path / "TF_x_BC_B_src_0_1_3mer", np.eye(len(weights), dtype=np.float32))
    model = qmodels.load_model(tmp_path / "TF_x_BC_B_src_0_1_3mer.weights.qbic")
    assert list(model.partial) == list(weights)
    cov = qmodels.load_covariance(tmp_path / f"TF_x_BC_B_src_0_1_3mer.cov_{len(weights)}.npy", len(weights))
    np.testing.assert_array_equal(cov, np.eye(len(weights)))


def test_sampling_caps_per_k():
    df = pd.DataFrame({"sequence": [f"s{i}" for i in range(10)], "score": range(10)})
    assert len(ols.sample_for_ols(df, 6, {6: None, 7: 4}, 42)) == 10
    sampled = ols.sample_for_ols(df, 7, {6: None, 7: 4}, 42)
    assert sampled.equals(df.sample(n=4, random_state=42))


def test_all_kmers_have_a_column():
    idx = kmers.kmer_index(3)
    assert set(idx) == {"".join(p) for p in product("ACGT", repeat=3)}


def test_ols_refuses_fewer_reads_than_features():
    df = pd.DataFrame({"sequence": ["ACGTACGTAC"] * 10, "score": range(10)})
    with pytest.raises(ValueError, match="more scored reads"):
        ols.train_kmer_model(df, 3, use_gpu=False)
