"""Score selection rule and the model-selection aggregation (Tables S2/S3)."""

import pandas as pd

from qbic_train.residual import select_scores
from selection.select_models import best_per_tf, top2_per_experiment


def test_alpha_rule():
    combined, residual = pd.DataFrame({"score": [1.0]}), pd.DataFrame({"score": [2.0]})
    assert select_scores(combined, residual, 0.3)[0] is residual      # alpha > 0: f_RES
    assert select_scores(combined, residual, 0.0)[0] is residual      # alpha = 0: f_RES (identical to f_SELEX)
    assert select_scores(combined, residual, -0.3)[0] is combined     # alpha < 0: f_SELEX


def _within():
    rows = []
    # experiment E1 of TF A: three models; m2 and m3 tie on the mean
    for model, rs in {"A_x_B1_b_s_0_1_6mer": [0.9, 0.7], "A_x_B1_b_s_0_2_6mer": [0.6, 0.6],
                      "A_x_B1_b_s_0_2_7mer": [0.5, 0.7]}.items():
        rows += [{"model_id": model, "target_pair_id": f"t{i}", "pearson_r": r} for i, r in enumerate(rs)]
    # experiment E2 of TF A: one model
    rows += [{"model_id": "A_x_B2_b_s_1_2_7mer", "target_pair_id": "t0", "pearson_r": 0.95}]
    return pd.DataFrame(rows)


def test_top2_per_experiment_keeps_first_on_ties():
    top = top2_per_experiment(_within())
    assert top.model_id.tolist() == ["A_x_B1_b_s_0_1_6mer", "A_x_B1_b_s_0_2_6mer", "A_x_B2_b_s_1_2_7mer"]
    assert top.within_exp_pearsonr.round(6).tolist() == [0.8, 0.6, 0.95]
    assert set(top.source_tf_label) == {"A"}


def test_best_per_tf(small_manifest):
    top = pd.DataFrame({"model_id": ["TFA_DBD_BC1_B1_yin2017_0_1_6mer", "TFA_DBD_BC1_B1_yin2017_0_2_7mer"],
                        "within_exp_pearsonr": [0.8, 0.7]})
    cross = pd.DataFrame({"model_id": ["TFA_DBD_BC1_B1_yin2017_0_1_6mer"] * 2 + ["TFA_DBD_BC1_B1_yin2017_0_2_7mer"] * 2,
                          "target_pair_id": ["p1", "p2", "p1", "p2"], "pearson_r": [0.5, 0.7, 0.65, 0.6]})
    best = best_per_tf(cross, top, small_manifest)
    assert best.model_id.tolist() == ["TFA_DBD_BC1_B1_yin2017_0_2_7mer"]
    row = best.iloc[0]
    assert row.cross_exp_pearsonr == 0.625 and row.within_exp_pearsonr == 0.7
    assert row.hgnc_symbol == "TFA" and row.ensembl_gene_id == "E1"
