"""Reading sequencing reads from common file formats (generic, any data source)."""

import gzip
from pathlib import Path

import pandas as pd
from Bio import SeqIO

_ACGT = frozenset("ACGT")


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_sequences(path, trim: int | None = None) -> list[str]:
    """Reads from FASTQ(.gz), FASTA(.gz), plain text (one read per line; a .txt file
    holding FASTQ records is read as FASTQ) or CSV (``sequence`` column).

    Reads containing anything other than A/C/G/T (e.g. N) are dropped, then
    reads are optionally cut to their first ``trim`` bases. Duplicate reads are
    kept, because the CNN is trained on all reads (Supplementary Methods).
    """
    path = Path(path)
    name = path.name.lower().removesuffix(".gz")
    if name.endswith((".fastq", ".fq")):
        with _open_text(path) as handle:
            seqs = [str(rec.seq) for rec in SeqIO.parse(handle, "fastq")]
    elif name.endswith((".fasta", ".fa", ".fna")):
        with _open_text(path) as handle:
            seqs = [str(rec.seq) for rec in SeqIO.parse(handle, "fasta")]
    elif name.endswith(".txt"):
        with _open_text(path) as handle:
            first = handle.readline()
        with _open_text(path) as handle:
            if first.startswith("@"):      # FASTQ content in a .txt(.gz) file (e.g. Yang et al. 2017)
                seqs = [str(rec.seq) for rec in SeqIO.parse(handle, "fastq")]
            else:                          # one read per line (e.g. Rodriguez-Martinez et al. 2017)
                seqs = [line.strip() for line in handle]
    elif name.endswith(".csv"):
        df = pd.read_csv(path)
        if "sequence" not in df.columns:
            raise ValueError(f"{path}: CSV input needs a 'sequence' column")
        seqs = df["sequence"].astype(str).tolist()
    else:
        raise ValueError(f"{path}: unsupported format (use FASTQ, FASTA, .txt or .csv, optionally .gz)")
    return clean_reads(seqs, trim)


def clean_reads(seqs, trim: int | None = None) -> list[str]:
    """Drop reads with non-ACGT characters, then optionally trim to ``trim`` bases."""
    seqs = [s for s in (s.upper() for s in seqs) if _ACGT.issuperset(s)]
    if trim:
        seqs = [s[:trim] for s in seqs]
    return seqs
