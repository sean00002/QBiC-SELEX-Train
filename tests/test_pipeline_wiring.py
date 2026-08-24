import argparse
import unittest
from unittest.mock import patch

import pandas as pd

from qbic_selex_train import run_single_experiment


class TestSelectedOutputPipelineWiring(unittest.TestCase):
    @patch("qbic_selex_train.os.path.exists", return_value=True)
    @patch("qbic_selex_train.OLSTrainer")
    @patch("qbic_selex_train.ResidualTrainer")
    @patch("qbic_selex_train.SelexDataExtractor")
    def test_ols_reads_selected_corrected_output(
        self,
        extractor_class,
        residual_trainer_class,
        ols_trainer_class,
        _exists,
    ):
        extractor_class.return_value.load_sequences_from_files.return_value = (
            ["AAAA"],
            ["CCCC"],
        )
        residual_trainer_class.return_value.train.return_value = pd.DataFrame(
            {"sequence": ["AAAA", "CCCC"], "score": [1.0, 2.0]}
        )

        args = argparse.Namespace(
            exp_id=None,
            positive_seqs="positive.txt",
            negative_seqs="negative.txt",
            control_round=0,
            enriched_round=1,
            stage=None,
            stages=None,
            bias_model="bias.pt",
            num=2,
            seed=42,
            save_uncorrected=False,
            save_models=False,
            kmer=6,
            mode=3,
        )
        config = {
            "data": {"fastqgz_path": "unused"},
            "bias_models": {"default": "bias.pt"},
            "residual_training": {"max_sequences": 2, "seed": 42},
            "ols_training": {"kmer_size": 6, "seed": 42, "mode": 3},
            "output": {"save_uncorrected": False, "save_models": False},
        }

        run_single_experiment(args, config)

        output_name = "sequences_"
        predictions_path = ols_trainer_class.return_value.train.call_args.args[0]
        self.assertTrue(predictions_path.startswith(
            "residual_model_output/corrected/" + output_name
        ))
        self.assertTrue(predictions_path.endswith("_0_1.csv"))


if __name__ == "__main__":
    unittest.main()
