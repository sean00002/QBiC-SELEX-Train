"""k-mer features for the OLS module (Methods: "Interpretable k-mer OLS model").

A k-mer and its reverse complement share one feature, stored under the
lexicographically larger of the two (palindromes once). This gives d = 2080
features for k = 6 and d = 8192 for k = 7, in sorted order: the same order as
the released ``*.weights.qbic`` files and their covariance matrices.
"""

from functools import cache
from itertools import product

import numpy as np

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def sliding_window(seq: str, k: int) -> list[str]:
    """All overlapping k-mers of ``seq``, left to right."""
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def canonical(kmer: str) -> str:
    """The stored representative of a k-mer: max(k-mer, reverse complement)."""
    return max(kmer, reverse_complement(kmer))


@cache
def canonical_kmers(k: int) -> tuple[str, ...]:
    """Sorted canonical k-mers: the feature order of the OLS model."""
    return tuple(sorted({canonical("".join(p)) for p in product("ACGT", repeat=k)}))


@cache
def kmer_index(k: int) -> dict[str, int]:
    """Column index of every k-mer (both orientations) in the feature matrix."""
    index = {kmer: i for i, kmer in enumerate(canonical_kmers(k))}
    return {"".join(p): index[canonical("".join(p))] for p in product("ACGT", repeat=k)}


def kmer_count_matrix(sequences, k: int) -> np.ndarray:
    """N x d matrix of canonical k-mer counts per sequence, as float32.

    Counts are small integers, so float32 holds them exactly; building the
    matrix in float32 directly gives the same values as counting in int64 and
    converting, with half the memory.
    """
    index = kmer_index(k)
    X = np.zeros((len(sequences), len(canonical_kmers(k))), dtype=np.float32)
    for row, seq in enumerate(sequences):
        for kmer in sliding_window(seq, k):
            X[row, index[kmer]] += 1
    return X


def expand_weights(weights: dict[str, float]) -> dict[str, float]:
    """Canonical-k-mer coefficients → coefficients for all 4^k k-mers."""
    full = dict(weights)
    for kmer, value in weights.items():
        full.setdefault(reverse_complement(kmer), value)
    return full


def score_sequences(weights: dict[str, float], sequences) -> np.ndarray:
    """Sum of k-mer weights over each sequence (all sequences of equal length).

    Builds the N x (L-k+1) float32 array of per-position weights and sums each
    row, the same computation the original validation script performed.
    """
    full = expand_weights(weights)
    k = len(next(iter(weights)))
    values = np.array([[full[kmer] for kmer in sliding_window(s, k)] for s in sequences], dtype=np.float32)
    return np.sum(values, axis=1)
