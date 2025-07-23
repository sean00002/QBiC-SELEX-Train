# CLAUDE.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

QBiC-SELEX-Train is a machine learning pipeline for training models on SELEX (Systematic Evolution of Ligands by Exponential enrichment) data to predict protein-DNA binding. The codebase implements a two-stage approach: bias model training followed by transcription factor-specific residual model training.

## Main Components

### Training Scripts
- `qbic_train_cnn_tf_model.py` - Main CNN training script that uses PyTorch to train residual models for transcription factors, outputs corrected/uncorrected prediction DataFrames
- `bias_model/qbic_train_cnn_bias_model.py` - CNN bias model training with hyperparameter optimization
- `qbic_train_ols_from_dfs.py` - Ordinary Least Squares model training from CNN-generated DataFrames (trains on corrected CNN predictions)

### Utility Scripts
- `util_scripts/selex_meta_utils.py` - SELEX experiment metadata handling and file discovery
- `util_scripts/sequence_utils.py` - DNA sequence processing, k-mer extraction, and one-hot encoding
- `util_scripts/metadata_experiments.csv` - Experiment metadata mapping

## Architecture

### Three-Stage Training Pipeline
1. **Bias Model**: Trains a general CNN to capture sequence-level biases across experiments
2. **Residual Model**: Trains TF-specific models that learn residuals on top of the frozen bias model, generates corrected/uncorrected prediction DataFrames
3. **OLS Model**: Trains linear models on k-mer features using the corrected CNN predictions as targets

### Model Types
- **UpdatedCNN**: Multi-layer CNN with batch normalization, dropout, and global max pooling
- **ResidualModel**: Combines frozen bias model with learnable residual branch and bias scale parameter
- **OLS Models**: Linear models trained on k-mer features using CNN-corrected predictions as regression targets

### Key Features
- One-hot encoding for DNA sequences (A=0, C=1, G=2, T=3)
- Reverse complement k-mer handling to reduce feature dimensionality
- CUDA/GPU acceleration with CuPy and CuML
- Cross-validation for hyperparameter optimization
- Early stopping with validation monitoring

## Common Commands

### Training Bias Model
```bash
python bias_model/qbic_train_cnn_bias_model.py -i <experiment_id> -e <enriched_round> -c <control_round> -o <output_dir>
```

### Training TF Model
```bash
python qbic_train_cnn_tf_model.py -i <exp_id/pair_id> -o <output_dir> -m <bias_model_path> -n <max_sequences> -s <seed>
```

### Training OLS Model
```bash
python qbic_train_ols_from_dfs.py -i <input_csv> -o <output_dir> -k <kmer_size> -n <max_sequences> -s <seed> -m <mode>
```

## File Structure and Data Flow

### Input Data
- SELEX experiment files stored in `/usr/project/xtmp/sl548/selex/selex_fastqgz_files/`
- Metadata in `util_scripts/metadata_experiments.csv` maps experiments to files
- Sequences are loaded from FASTQ.gz files and processed to remove 'N' nucleotides

### Output Structure
```
output_dir/
├── dfs/           # DataFrames with model predictions
├── cnn_models/    # Trained PyTorch models (.pt files)
├── ols_models/    # OLS coefficient dictionaries (.qbic files)
├── ols_covs/      # Covariance matrices (.npy files)
└── parameters/    # Hyperparameter search results
```

### Model Checkpoints
- CNN models save state_dict, configuration, and performance metrics
- OLS models save coefficient dictionaries as pickled files
- Covariance matrices stored as compressed upper-triangle numpy arrays

## Dependencies

The codebase requires:
- PyTorch for neural network training
- CuPy/CuML for GPU-accelerated linear algebra
- BioPython for FASTQ file parsing
- pandas, numpy, scipy for data processing
- scikit-learn for model evaluation metrics

## Data Processing Pipeline

1. **Experiment Discovery**: Use metadata utils to find FASTQ files for given experiment IDs
2. **Sequence Loading**: Extract sequences from FASTQ files, filter out 'N' nucleotides
3. **Bias Model Training**: Train general CNN on raw SELEX data (control vs enriched)
4. **Residual Model Training**: Train TF-specific CNN using frozen bias model, output corrected/uncorrected prediction DataFrames
5. **OLS Model Training**: Train linear models on k-mer features using corrected CNN predictions as regression targets
6. **Evaluation**: Save all models with performance metrics and predictions

## Special Handling

### Bhimsaria2023 Experiments
- Sequences truncated to first 20 base pairs
- Special file path handling in `/usr/project/xtmp/sl548/selex/selex_fastqgz_files/bhimsaria2023/`
- Different round numbering scheme including zero rounds

### GPU Memory Management
- Automatic GPU selection for multi-GPU systems
- Memory cleanup on CUDA out-of-memory errors Batch size adjustment based on available memory
