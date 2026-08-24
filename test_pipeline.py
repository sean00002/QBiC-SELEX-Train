#!/usr/bin/env python3

import os
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
import torch
from pandas.testing import assert_frame_equal
from modules.selex_data_extractor import SelexDataExtractor
from modules.residual_trainer import ResidualTrainer
from modules.ols_trainer import OLSTrainer

def create_test_data():
    """Create small test dataset"""
    np.random.seed(42)
    
    # Generate synthetic DNA sequences
    bases = ['A', 'C', 'G', 'T']
    control_seqs = []
    enriched_seqs = []
    
    # Control sequences (100 sequences, 20bp each)
    for _ in range(100):
        seq = ''.join(np.random.choice(bases, 20))
        control_seqs.append(seq)
    
    # Enriched sequences (100 sequences, 20bp each)
    for _ in range(100):
        seq = ''.join(np.random.choice(bases, 20))
        enriched_seqs.append(seq)
    
    return control_seqs, enriched_seqs

def test_residual_trainer():
    """Test residual trainer module"""
    print("Testing residual trainer...")
    
    # Create test data
    control_seqs, enriched_seqs = create_test_data()
    
    # Check if bias model exists
    bias_model_path = "bias_model/beadOnlyControl_NA_TG40NCATATG_AATBA_jolma2025_0_2_CNN11K128B512F_L2_DTrue.pt"
    if not os.path.exists(bias_model_path):
        print(f"Warning: Bias model not found at {bias_model_path}")
        print("Skipping residual trainer test")
        return
    
    # Test residual trainer
    config = {'residual_training': {'max_sequences': 200, 'seed': 42}}
    trainer = ResidualTrainer(config)
    
    try:
        corrected_df = trainer.train(
            control_seqs, enriched_seqs, bias_model_path, 
            "test_output", num=200, seed=42, save_uncorrected=True, save_model=True
        )
        
        # Verify outputs
        assert os.path.exists("residual_model_output/corrected/test_output.csv")
        assert os.path.exists("residual_model_output/uncorrected/test_output.csv")
        assert os.path.exists("residual_model_output/metadata/test_output.json")
        assert os.path.exists("models/residual/test_output.pt")

        saved_selected_df = pd.read_csv(
            "residual_model_output/corrected/test_output.csv"
        )
        assert_frame_equal(
            corrected_df.reset_index(drop=True),
            saved_selected_df.reset_index(drop=True),
            check_dtype=False,
        )

        with open("residual_model_output/metadata/test_output.json") as f:
            metadata = json.load(f)
        checkpoint = torch.load(
            "models/residual/test_output.pt",
            map_location="cpu",
            weights_only=False,
        )

        expected_selected_score = (
            "f_RES" if checkpoint["bias_scale"] >= 0 else "f_SELEX"
        )
        assert checkpoint["selected_score"] == expected_selected_score
        assert metadata["selected_score"] == expected_selected_score
        assert metadata["alpha"] == checkpoint["bias_scale"]
        assert metadata["selection_rule"] == (
            "f_RES if alpha >= 0 else f_SELEX"
        )
        
        # Check DataFrame structure
        assert 'sequence' in corrected_df.columns
        assert 'score' in corrected_df.columns
        assert len(corrected_df) <= 200
        
        print("✓ Residual trainer test passed")
        
    except Exception as e:
        print(f"✗ Residual trainer test failed: {e}")
        raise

def test_ols_trainer():
    """Test OLS trainer module"""
    print("Testing OLS trainer...")
    
    # Create test predictions CSV
    test_seqs, _ = create_test_data()
    test_scores = np.random.randn(len(test_seqs))
    
    test_df = pd.DataFrame({
        'sequence': test_seqs,
        'score': test_scores
    })
    
    os.makedirs("test_data", exist_ok=True)
    test_csv = "test_data/test_predictions.csv"
    test_df.to_csv(test_csv, index=False)
    
    # Test OLS trainer
    config = {'ols_training': {'seed': 42}}
    trainer = OLSTrainer(config)
    
    try:
        # Use a small feature space so this smoke test remains full rank with
        # the synthetic dataset. Production training still uses 6- or 7-mers.
        lm_dic = trainer.train(test_csv, "test_ols_output", kmer_size=3, num=100, seed=42, mode=3)
        
        # Verify outputs
        assert os.path.exists("k_mer_weights/test_ols_output_3mer.qbic")
        assert any(f.startswith("test_ols_output_3mer.cov_") for f in os.listdir("covariance_matrices/"))
        
        # Check coefficients
        assert isinstance(lm_dic, dict)
        assert len(lm_dic) > 0
        
        print("✓ OLS trainer test passed")
        
    except Exception as e:
        print(f"✗ OLS trainer test failed: {e}")
        raise

def test_data_extractor():
    """Test data extractor module"""
    print("Testing data extractor...")
    
    # Create test sequence files
    os.makedirs("test_data", exist_ok=True)
    
    # Test txt format
    with open("test_data/positive.txt", 'w') as f:
        for seq in ["ATCGATCGATCGATCGATCG", "GCTAGCTAGCTAGCTAGCTA"]:
            f.write(f"{seq}\n")
    
    with open("test_data/negative.txt", 'w') as f:
        for seq in ["AAAAAAAAAAAAAAAAAAA", "TTTTTTTTTTTTTTTTTTTT"]:
            f.write(f"{seq}\n")
    
    # Test CSV format
    test_df = pd.DataFrame({
        'sequence': ["ATCGATCGATCGATCGATCG", "GCTAGCTAGCTAGCTAGCTA"]
    })
    test_df.to_csv("test_data/test_seqs.csv", index=False)
    
    extractor = SelexDataExtractor()
    
    try:
        # Test file loading
        control_seqs, enriched_seqs = extractor.load_sequences_from_files(
            "test_data/positive.txt", "test_data/negative.txt"
        )
        
        assert len(control_seqs) == 2
        assert len(enriched_seqs) == 2
        assert "ATCGATCGATCGATCGATCG" in enriched_seqs
        
        print("✓ Data extractor test passed")
        
    except Exception as e:
        print(f"✗ Data extractor test failed: {e}")
        raise

def cleanup():
    """Clean up test files"""
    dirs_to_remove = [
        "residual_model_output", "models", "k_mer_weights", 
        "covariance_matrices", "test_data"
    ]
    
    for dirname in dirs_to_remove:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

def main():
    print("Running QBiC pipeline tests...")
    print("=" * 50)
    
    # Clean up any existing test files
    cleanup()
    
    # Run tests
    test_data_extractor()
    test_residual_trainer()
    test_ols_trainer()
    
    print("=" * 50)
    print("Test completed. Cleaning up...")
    
    # Clean up test files
    cleanup()
    
    print("Done.")

if __name__ == "__main__":
    main()
