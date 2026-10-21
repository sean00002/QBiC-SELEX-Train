"""Paired-cycle rule (Methods) and Table S1 lookups."""

from qbic_data import manifest as mf


def test_methods_example_gives_nine_pairs_in_training_order(small_manifest):
    pairs = mf.experiment_pairs("TFA_DBD_BC1_B1_yin2017", small_manifest)
    assert pairs == [("0", "1"), ("0", "2"), ("0", "3"), ("0", "4"), ("1", "2"), ("1", "3"), ("1", "4"),
                     ("2", "3"), ("2", "4")]


def test_cycle_zero_comes_from_the_matching_library(small_manifest):
    files = mf.experiment_files("TFA_DBD_BC1_B1_yin2017", small_manifest)
    assert [f.cycle for f in files] == ["0", "1", "2", "3", "4"]
    assert mf.cycle_file("TFA_DBD_BC1_B1_yin2017", "0", small_manifest).filename == "zero.fastq.gz"


def test_bhimsaria_lib1_example_gives_twelve_pairs(small_manifest):
    pairs = mf.experiment_pairs("TLX_NA_Lib1_NA_bhimsaria2023", small_manifest)
    assert len(pairs) == 12
    assert pairs[:3] == [("0(3)", "1"), ("0(3)", "2"), ("0(3)", "3")]
    assert pairs[-3:] == [("1", "2"), ("1", "3"), ("2", "3")]


def test_ids_parse():
    assert mf.parse_pair_id("TLX_NA_Lib1_NA_bhimsaria2023_0(2)_2_7mer") == ("TLX_NA_Lib1_NA_bhimsaria2023", "0(2)", "2")
    assert mf.parse_pair_id("ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4") == ("ETV4_eDBD_TTTGCC40NTGA_KS_yin2017", "0", "4")
    assert mf.parse_exp_id("ETV4_eDBD_TTTGCC40NTGA_KS_yin2017")["barcode"] == "TTTGCC40NTGA"


def test_real_table_s1_examples():
    mf.load_manifest.cache_clear()
    etv4 = "ETV4_eDBD_TTTGCC40NTGA_KS_yin2017"
    assert len(mf.experiment_pairs(etv4)) == 9
    assert mf.cycle_file(etv4, "0").filename == "ZeroCycle_TTTGCC40NTGA_0_0.fastq.gz"
    assert ("0(2)", "2") in mf.experiment_pairs("TLX_NA_Lib1_NA_bhimsaria2023")
    m = mf.load_manifest()
    assert len(m) == 19027 and m.exp_id.nunique() == 6385


def test_files_for_pairs_dedupes(small_manifest):
    files = mf.files_for_pairs([("TFA_DBD_BC1_B1_yin2017", "0", "4"), ("TFA_DBD_BC1_B1_yin2017", "0", "3")],
                               small_manifest)
    assert [f.filename for f in files] == ["zero.fastq.gz", "a4.fastq.gz", "a3.fastq.gz"]


def test_cli_trains_each_pair_of_an_experiment_separately():
    import qbic_selex_train as cli
    from qbic_train.config import load_config
    mf.load_manifest.cache_clear()
    exp = "ETV4_eDBD_TTTGCC40NTGA_KS_yin2017"
    pairs, _, needed = cli.build_jobs(cli.parse_args(["--exp_id", exp]), load_config())
    assert pairs == [f"{exp}_{e}_{c}" for e, c in mf.experiment_pairs(exp)] and len(pairs) == 9
    pairs, _, _ = cli.build_jobs(cli.parse_args(["--model_id", f"{exp}_0_4_7mer"]), load_config())
    assert pairs == [f"{exp}_0_4"]
