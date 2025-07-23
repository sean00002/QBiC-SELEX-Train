# QBiC-SELEX-Train

QBiC-SELEX pipeline for training bias-corrected models on HT-SELEX data to predict variant effects on TF binding.

## Overview

This pipeline implements a two-stage approach for SELEX analysis:
1. **Residual Model Training**: CNN-based model using bias correction to classify a pair of control and enriched sequences
2. **OLS Model Training**: Linear model on k-mer features using CNN residual model corrected outputs

The default bias model curated by us is in `/bias_model/`, and users can train and use their own bias model based on control TF-free HT-SELEX data. 

## Quick Start

### Single Experiment
```bash
# Full pipeline with experiment ID
python qbic_selex_train.py --exp_id ELK1_FL_TG40NCATATG_KT_jolma2013 --control_round 0 --enriched_round 2

# Using sequence files directly
python qbic_selex_train.py --positive_seqs enriched.txt --negative_seqs control.txt
```

### Batch Processing
```bash
# Create batch file (CSV format)
echo "exp_id,control_round,enriched_round" > experiments.txt
echo "ELK1_FL_TG40NCATATG_KT_jolma2013,0,2" >> experiments.txt

# Run batch
python qbic_selex_train.py --batch experiments.txt
```

### Individual Stages
```bash
# Only residual training
python qbic_selex_train.py --stage residual --positive_seqs pos.txt --negative_seqs neg.txt

# Only OLS training (requires existing predictions)
python qbic_selex_train.py --stage ols --exp_id ELK1_D00_TG40NCATATG_AATBA_jolma2025
```

## Output Structure

```
output/
├── residual_model_output/
│   ├── corrected/           # CNN corrected predictions (default)
│   └── uncorrected/         # CNN uncorrected predictions (optional)
├── k_mer_weights/           # OLS coefficients (default)
├── covariance_matrices/     # Covariance matrices (default)
├── models/                  # Trained models (optional)
│   └── residual/
└── logs/
    └── failed_experiments.txt
```

## Configuration

Edit `config/config.yaml` to customize:
- Bias model paths
- Training parameters
- Default k-mer sizes
- Output settings

## Options

### Core Parameters
- `--exp_id`: Experiment identifier
- `--control_round`, `--enriched_round`: Round numbers
- `--positive_seqs`, `--negative_seqs`: Sequence files
- `--batch`: Batch processing file

### Model Options
- `--bias_model`: Custom bias model path
- `--kmer`: K-mer size (6, 7, or 8)
- `--mode`: Output mode (1=weights, 2=covariance, 3=both)

### Training Parameters
- `--num`: Max sequences to use
- `--seed`: Random seed

### Output Control
- `--save_uncorrected`: Save uncorrected predictions
- `--save_models`: Save trained CNN models

### Stage Control
- `--stage`: Run single stage (extract, residual, ols)
- `--stages`: Run multiple stages (comma-separated)

## Input Formats

Sequence files support:
- **TXT**: One sequence per line
- **FASTA**: Standard FASTA format  
- **CSV**: Must contain 'sequence' column

## Batch File Format

CSV with columns:
- `exp_id` (required)
- `control_round`, `enriched_round` (optional, uses CLI defaults)
- `seed`, `kmer_size`, `bias_model` (optional overrides)

Example:
```csv
exp_id,control_round,enriched_round,seed,kmer_size
ELK1_FL_TG40NCATATG_KT_jolma2013,0,2,42,6
```

## Dependencies

### Required Dependencies
```bash
pip install pandas==1.5.3 numpy==1.23.5 torch==2.4.0.post301 scikit-learn==1.2.2 biopython==1.81 pyyaml==6.0.1 tqdm==4.67.1
```

- **pandas** (1.5.3)
- **numpy** (1.23.5)
- **torch** (2.4.0)
- **scikit-learn** (1.2.2)
- **biopython** (1.81)
- **pyyaml** (6.0.1)
- **cudf** (23.12.1)
- **cuml** (23.12.0)
- **cupy** (13.0.0)

## Testing

Run test suite:
```bash
python test_pipeline.py
```

