"""Read loading and source-specific preparation."""

import gzip

import pytest

from qbic_data import manifest as mf
from qbic_train.reads import read_sequences


def _fastq(seqs):
    return "".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(seqs))


def test_formats_and_n_filter(tmp_path):
    seqs = ["ACGTACGT", "ACGNACGT", "acgtacgt", "TTTTAAAA"]
    expected = ["ACGTACGT", "ACGTACGT", "TTTTAAAA"]   # N dropped, lowercase upper-cased, duplicates kept
    with gzip.open(tmp_path / "r.fastq.gz", "wt") as f:
        f.write(_fastq(seqs))
    (tmp_path / "r.fastq").write_text(_fastq(seqs))
    (tmp_path / "r.fa").write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))
    (tmp_path / "r.txt").write_text("\n".join(seqs) + "\n")
    (tmp_path / "r.csv").write_text("sequence\n" + "\n".join(seqs) + "\n")
    for name in ("r.fastq.gz", "r.fastq", "r.fa", "r.txt", "r.csv"):
        assert read_sequences(tmp_path / name) == expected, name
    assert read_sequences(tmp_path / "r.txt", trim=4) == ["ACGT", "ACGT", "TTTT"]
    with pytest.raises(ValueError):
        read_sequences(tmp_path / "r.bam")


def test_bhimsaria_reads_are_cut_to_20bp_after_n_filter(tmp_path, small_manifest):
    d = tmp_path / "reads" / "bhimsaria2023"
    d.mkdir(parents=True)
    long_ok, long_n = "A" * 20 + "C" * 10, "A" * 20 + "N" + "C" * 9   # N outside the first 20 bp still drops the read
    with gzip.open(d / "t2.fastq.gz", "wt") as f:
        f.write(_fastq([long_ok, long_n]))
    reads = mf.load_cycle_reads("TLX_NA_Lib1_NA_bhimsaria2023", "2", tmp_path / "reads", small_manifest)
    assert reads == ["A" * 20]


def test_missing_reads_explain_how_to_download(tmp_path, small_manifest):
    with pytest.raises(FileNotFoundError, match="download_reads.py"):
        mf.load_cycle_reads("TFA_DBD_BC1_B1_yin2017", "1", tmp_path, small_manifest)
