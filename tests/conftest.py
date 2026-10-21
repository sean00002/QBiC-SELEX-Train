import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def small_manifest(tmp_path):
    """A two-experiment Table S1 excerpt: the Methods example (cycles 1-4 plus a
    cycle-0 library) and the Bhimsaria Lib1 example (three bead-only selections)."""
    cols = "tf,source_tf_label,hgnc_symbol,ensembl_gene_id,gene_mapping_status,record_class,type,dbd,barcode,batch,cycle,source,exp_id,cycle_id,study_accession,filename_ftp,submitted_ftp"
    rows = [
        "ZEROCYCLE,,,,,cycle-zero,t,NA,BC1,NA,0,yin2017,ZEROCYCLE_NA_BC1_NA_yin2017,x,P,zero.fastq.gz,ftp.sra.ebi.ac.uk/zero.fastq.gz",
        *[f"TFA,TFA,TFA,E1,x,tf,t,DBD,BC1,B1,{c},yin2017,TFA_DBD_BC1_B1_yin2017,x,P,a{c}.fastq.gz,ftp.sra.ebi.ac.uk/a{c}.fastq.gz"
          for c in (4, 2, 1, 3)],
        *[f"ZEROCYCLE,,,,,cycle-zero,t,NA,Lib1,NA,0({i}),bhimsaria2023,ZEROCYCLE_NA_Lib1_NA_bhimsaria2023,x,P,z{i}.fastq.gz,prefetch SRR{i} fastq-dump --gzip"
          for i in (3, 2, 1)],
        *[f"TLX,TLX,NR2E1,E2,x,tf,t,NA,Lib1,NA,{c},bhimsaria2023,TLX_NA_Lib1_NA_bhimsaria2023,x,P,t{c}.fastq.gz,prefetch SRT{c} fastq-dump --gzip"
          for c in (3, 1, 2)],
    ]
    path = tmp_path / "TableS1.csv"
    path.write_text(cols + "\n" + "\n".join(rows) + "\n")
    from qbic_data.manifest import load_manifest
    load_manifest.cache_clear()
    return str(path)
