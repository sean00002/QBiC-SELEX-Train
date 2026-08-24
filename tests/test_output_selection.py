import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from modules.residual_trainer import ResidualTrainer


class TestAlphaDependentOutputSelection(unittest.TestCase):
    def setUp(self):
        self.combined_df = pd.DataFrame(
            {"sequence": ["AAAA", "CCCC"], "score": [2.0, 3.0]}
        )
        self.residual_df = pd.DataFrame(
            {"sequence": ["AAAA", "CCCC"], "score": [1.0, 1.5]}
        )

    def test_positive_alpha_uses_residual_as_selected_score(self):
        selected_df, alternate_df = ResidualTrainer._select_output_dataframes(
            self.combined_df,
            self.residual_df,
            0.25,
        )

        assert_frame_equal(selected_df, self.residual_df)
        assert_frame_equal(alternate_df, self.combined_df)

    def test_negative_alpha_uses_combined_as_selected_score(self):
        selected_df, alternate_df = ResidualTrainer._select_output_dataframes(
            self.combined_df,
            self.residual_df,
            -0.25,
        )

        assert_frame_equal(selected_df, self.combined_df)
        assert_frame_equal(alternate_df, self.residual_df)

    def test_zero_alpha_uses_nonnegative_branch(self):
        selected_df, alternate_df = ResidualTrainer._select_output_dataframes(
            self.combined_df,
            self.residual_df,
            0.0,
        )

        assert_frame_equal(selected_df, self.residual_df)
        assert_frame_equal(alternate_df, self.combined_df)


if __name__ == "__main__":
    unittest.main()
