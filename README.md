# QBiC-SELEX-Train

QBiC-SELEX pipeline for training bias-corrected models on HT-SELEX data to predict variant effects on TF binding.

## Overview

QBiC-SELEX has three conceptual model components:

1. **Universal bias model**: a pretrained CNN that captures shared
   SELEX-associated sequence bias.
2. **TF-specific residual model**: a CNN residual branch trained with the
   frozen universal bias model and a learned bias-scaling coefficient.
3. **k-mer OLS model**: a linear model trained on the alpha-dependent CNN score
   selected by the residual-training stage.

The public command-line workflow runs residual-model training followed by OLS
training when a pretrained universal bias model is supplied.

The default bias model curated by us is in `/bias_model/`, and users can train and use their own bias model based on control TF-free HT-SELEX data. 

## Score semantics

For an input sequence `x`, let `f_BIAS(x)` be the frozen universal bias-model
logit, `f_RES(x)` the TF-specific residual-branch logit, and
`alpha = tanh(raw_bias_scale)`. The complete early-versus-late classifier is:

```text
f_SELEX(x) = f_RES(x) + alpha * f_BIAS(x)
```

The score selected for downstream QBiC OLS training is:

```text
f_QBiC(x) = f_RES(x)   if alpha >= 0
            f_SELEX(x) if alpha < 0
```

At `alpha = 0`, the two scores are numerically identical. The phrase
**selected QBiC score** is used below because the historical directory name
`corrected/` does not always correspond to the residual branch.

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

```text
.
├── residual_model_output/
│   ├── corrected/           # Alpha-dependent score selected for OLS
│   ├── uncorrected/         # Alternate CNN-derived score (optional)
│   └── metadata/            # Alpha and selected-score semantics
├── models/
│   └── residual/            # Trained residual checkpoints (optional)
├── k_mer_weights/           # OLS coefficients (default)
├── covariance_matrices/     # Covariance matrices (default)
└── logs/
    └── failed_experiments.txt
```

The directory names are retained for backward compatibility:

```text
residual_model_output/corrected/
    Selected QBiC score used for OLS training.
    alpha >= 0: f_RES
    alpha < 0:  f_SELEX

residual_model_output/uncorrected/
    Alternate CNN-derived score saved for diagnostics when requested.
```

The corrected-versus-uncorrected analysis described in the QBiC-SELEX
manuscript is restricted to models with `alpha > 0`. Within that subset,
`corrected = f_RES` and `uncorrected = f_SELEX`.

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
- `--save_uncorrected`: Save the alternate CNN-derived predictions
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
python -m unittest discover -s tests

# Optional end-to-end training test (uses the bundled bias-model checkpoint)
python test_pipeline.py
```

## Compatibility note

CNN score files and downstream OLS artifacts generated by modular public
pipeline versions before the alpha-selection fix may be mislabeled. Regenerate
the CNN scores, OLS coefficients, and covariance matrices with the corrected
pipeline unless the sign of alpha and the identity of both saved outputs are
known. Production model collections generated with the original audited
training scripts already used the intended selection rule and are not affected
solely by this modular refactoring regression.
