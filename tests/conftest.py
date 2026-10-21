import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def small_manifest(tmp_path):
    """A two-experiment Table S1 excerpt: the Methods example (cycles 1-4 plus a
    cycle-0 library) and the Bhimsaria Lib1 example (three bead-only selections)."""
    cols = ["tf", "source_tf_label", "hgnc_symbol", "ensembl_gene_id", "gene_mapping_status", "record_class",
            "type", "dbd", "barcode", "batch", "cycle", "source", "exp_id", "cycle_id", "study_accession",
            "filename_ftp", "submitted_ftp"]

    def row(tf, label, hgnc, ens, rclass, dbd, bc, batch, cycle, source, exp, fname, ftp):
        return ",".join([tf, label, hgnc, ens, "x", rclass, "t", dbd, bc, batch, cycle, source, exp, "x", "P",
                         fname, ftp])

    rows = [row("ZEROCYCLE", "", "", "", "cycle-zero", "NA", "BC1", "NA", "0", "yin2017",
                "ZEROCYCLE_NA_BC1_NA_yin2017", "zero.fastq.gz", "ftp.sra.ebi.ac.uk/zero.fastq.gz")]
    rows += [row("TFA", "TFA", "TFA", "E1", "tf", "DBD", "BC1", "B1", str(c), "yin2017", "TFA_DBD_BC1_B1_yin2017",
                 f"a{c}.fastq.gz", f"ftp.sra.ebi.ac.uk/a{c}.fastq.gz") for c in (4, 2, 1, 3)]
    rows += [row("ZEROCYCLE", "", "", "", "cycle-zero", "NA", "Lib1", "NA", f"0({i})", "bhimsaria2023",
                 "ZEROCYCLE_NA_Lib1_NA_bhimsaria2023", f"z{i}.fastq.gz", f"prefetch SRR{i} fastq-dump --gzip")
             for i in (3, 2, 1)]
    rows += [row("TLX", "TLX", "NR2E1", "E2", "tf", "NA", "Lib1", "NA", str(c), "bhimsaria2023",
                 "TLX_NA_Lib1_NA_bhimsaria2023", f"t{c}.fastq.gz", f"prefetch SRT{c} fastq-dump --gzip")
             for c in (3, 1, 2)]
    path = tmp_path / "TableS1.csv"
    path.write_text(",".join(cols) + "\n" + "\n".join(rows) + "\n")
    from qbic_data.manifest import load_manifest
    load_manifest.cache_clear()
    return str(path)
