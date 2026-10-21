"""Table S1 lookups: which read file holds each cycle of an experiment, which
cycle pairs are trained, and how each source's reads are prepared.

Rules (Methods: "Paired-cycle design"; Supplementary Methods: "HT-SELEX data
preprocessing"):
* Cycle 0 of an experiment is the initial library (``record_class ==
  cycle-zero``) with the same barcode and source.
* Pairs: early cycle 0, 1 or 2; late cycle = any later cycle.
* Bhimsaria et al. (2023): the bead-only library selections 0(1), 0(2), 0(3)
  are extra early cycles; reads are cut to their first 20 bp.
* Rodriguez-Martinez et al. (2017): text files that already hold the 20 bp
  random region.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from qbic_train.config import REPO_ROOT
from qbic_train.reads import read_sequences

DEFAULT_MANIFEST = REPO_ROOT / "data" / "TableS1_metadata_experiments.csv"
BHIMSARIA = "bhimsaria2023"
BHIMSARIA_TRIM = 20


@dataclass(frozen=True)
class CycleFile:
    exp_id: str
    cycle: str          # "0", "1", ... or "0(1)" for Bhimsaria bead-only selections
    source: str
    filename: str
    submitted_ftp: str

    def path(self, reads_dir) -> Path:
        return Path(reads_dir) / self.source / self.filename


@lru_cache(maxsize=4)
def load_manifest(path=None) -> pd.DataFrame:
    return pd.read_csv(path or DEFAULT_MANIFEST, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _row_to_file(row) -> CycleFile:
    return CycleFile(row.exp_id, row.cycle, row.source, row.filename_ftp, row.submitted_ftp)


def parse_exp_id(exp_id: str) -> dict:
    """``TF_DBD_BARCODE_BATCH_SOURCE`` -> its parts."""
    parts = exp_id.split("_")
    if len(parts) != 5:
        raise ValueError(f"not an experiment ID (expected 5 fields): {exp_id}")
    return dict(zip(["tf", "dbd", "barcode", "batch", "source"], parts, strict=True))


def parse_pair_id(pair_id: str) -> tuple[str, str, str]:
    """``<exp_id>_<early>_<late>`` (an optional ``_<k>mer`` suffix is ignored)."""
    parts = pair_id.split("_")
    if len(parts) >= 8 and parts[7].endswith("mer"):
        parts = parts[:7]
    if len(parts) != 7:
        raise ValueError(f"not a pair or model ID: {pair_id}")
    return "_".join(parts[:5]), parts[5], parts[6]


def experiment_files(exp_id: str, manifest=None) -> list[CycleFile]:
    """All cycle files of an experiment: initial-library (cycle 0) files first,
    then the experiment's own cycles in increasing order."""
    m = load_manifest(manifest)
    info = parse_exp_id(exp_id)
    own = m[m.exp_id == exp_id]
    if own.empty:
        raise KeyError(f"experiment not in Table S1: {exp_id}")
    zero = m[(m.record_class == "cycle-zero") & (m.source == info["source"]) & (m.barcode == info["barcode"])]
    if info["source"] == BHIMSARIA:
        own = own.sort_values("cycle", kind="stable")
    else:
        own = own.assign(_c=own.cycle.astype(int)).sort_values("_c", kind="stable")
    return [_row_to_file(r) for r in zero.itertuples()] + [_row_to_file(r) for r in own.itertuples()]


def _is_zero(cycle: str) -> bool:
    return cycle == "0" or cycle.startswith("0(")


def experiment_pairs(exp_id: str, manifest=None) -> list[tuple[str, str]]:
    """Cycle pairs of an experiment, in the order the original code trained them."""
    files = experiment_files(exp_id, manifest)
    if parse_exp_id(exp_id)["source"] == BHIMSARIA:
        zero = [f.cycle for f in files if _is_zero(f.cycle)]
        later = [int(f.cycle) for f in files if not _is_zero(f.cycle)]
        pairs = [(z, str(c)) for z in zero for c in later]
        pairs += [(str(e), str(c)) for e in later if e <= 2 for c in later if c > e]
        return pairs
    cycles = [0 if _is_zero(f.cycle) else int(f.cycle) for f in files]
    return [(str(e), str(c)) for e in cycles if e <= 2 for c in cycles if c > e]


def cycle_file(exp_id: str, cycle: str, manifest=None) -> CycleFile:
    for f in experiment_files(exp_id, manifest):
        if f.cycle == str(cycle) or (str(cycle) == "0" and f.cycle == "0"):
            return f
    raise KeyError(f"{exp_id} has no cycle {cycle}")


def load_cycle_reads(exp_id: str, cycle: str, reads_dir, manifest=None) -> list[str]:
    """Reads of one cycle, prepared with the source's rules."""
    f = cycle_file(exp_id, cycle, manifest)
    path = f.path(reads_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; download it with: python download_reads.py --exp_id {exp_id}")
    return read_sequences(path, trim=BHIMSARIA_TRIM if f.source == BHIMSARIA else None)


def files_for_pairs(pairs, manifest=None) -> list[CycleFile]:
    """Unique cycle files needed for (exp_id, early, late) pairs."""
    seen, out = set(), []
    for exp_id, early, late in pairs:
        for cycle in (early, late):
            f = cycle_file(exp_id, cycle, manifest)
            if (f.source, f.filename) not in seen:
                seen.add((f.source, f.filename))
                out.append(f)
    return out
