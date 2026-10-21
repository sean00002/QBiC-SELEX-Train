# QBiC-SELEX-Train

Training code for **QBiC-SELEX**, the bias-corrected models of variant effects on transcription
factor (TF) binding described in *QBiC-SELEX: Quantitative, Bias-Corrected Modeling of Variant
Effects on Transcription Factor Binding* (Li & Gordân).

This repository trains the models from raw HT-SELEX reads and selects the published collection.
To **use** the trained models (variant-effect prediction), install
[QBiC-SELEX](https://github.com/sean00002/QBiC-SELEX) (`pip install qbic-selex`); the models
themselves are on Zenodo ([10.5281/zenodo.23482275](https://doi.org/10.5281/zenodo.23482275)).
The files written here use the same formats, so `qbic-selex` reads them directly.

## Contents

| File or folder | What it does |
|---|---|
| `qbic_selex_train.py` | Command line: train TF models (residual CNN, then 6- and 7-mer OLS) for cycle pairs |
| `qbic_train_bias_model.py` | Command line: train a universal bias-model candidate (grid search, then final fit) |
| `download_reads.py` | Command line: download the paper's raw reads (files listed in Table S1) |
| `make_pairs.py` | Command line: list the cycle pairs of experiments, as a CSV for `--batch` |
| `qbic_train/networks.py` | Network architectures: the bias CNN, the residual branch, the combined model f_SELEX = α·f_BIAS + f_RES, one-hot encoding |
| `qbic_train/residual.py` | Trains the residual model on early vs late reads, scores the reads, and picks f_QBiC (f_RES or f_SELEX by α) |
| `qbic_train/kmers.py` | k-mer features: the 2,080 / 8,192 reverse-complement-merged k-mers, the count matrix, summing k-mer weights over a sequence |
| `qbic_train/ols.py` | OLS fit with its covariance; writes `.weights.qbic` and `.cov_<n>.npy` files |
| `qbic_train/bias_model.py` | Loads the bias model; grid search and final fit for new bias-model candidates |
| `qbic_train/validation.py` | Pearson r between a k-mer model's predictions and CNN scores |
| `qbic_train/pipeline.py` | Runs the steps for each pair and writes the outputs |
| `qbic_train/reads.py`, `seeding.py`, `config.py` | Read loading, seeding, configuration |
| `qbic_data/` | The paper's data: Table S1 lookups, cycle pairs, source-specific read preparation, downloads |
| `selection/` | Within- and cross-experiment validation, top-2 and best-model selection (Tables S2, S3), bias-candidate comparison |
| `config/config.yaml` | All hyperparameters |
| `bias_model/` | The universal bias model used in the paper |
| `data/TableS1_metadata_experiments.csv` | Table S1: experiments, cycles and read-file accessions |
| `workflow/` | End-to-end example and Slurm scripts |

## Installation

```bash
git clone https://github.com/sean00002/QBiC-SELEX-Train.git
cd QBiC-SELEX-Train
python -m venv .venv && source .venv/bin/activate
pip install torch            # pick the CUDA build for your GPU: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install qbic-selex       # optional: compare with the released models, predict variant effects
pip install cupy-cuda12x     # optional: OLS on the GPU, as in the paper (cupy-cuda13x for CUDA 13)
```

Python ≥ 3.10. A CUDA GPU is strongly recommended for training; everything also runs on CPU, slowly.
The paper's models were trained with Python 3.10, PyTorch 2.4.0, NumPy 1.23.5, pandas 1.5.3,
scikit-learn 1.2.2, Biopython 1.81 and CuPy 13.0. This version was checked with
Python 3.11, PyTorch 2.14, NumPy 2.4, pandas 3.0 and CuPy 14.2.

## Quick start: rebuild a released model from raw reads

```bash
bash workflow/run_example.sh
```

This downloads the five cycles of the ETV4 experiment `ETV4_eDBD_TTTGCC40NTGA_KS_yin2017`
(~45 MB from ENA), trains all nine cycle pairs,
runs the within-experiment validation, compares the `0_4` 7-mer model with the released
`ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer`, and predicts variant effects with it. It takes
about 45 minutes on one GPU. The steps, one by one:

```bash
python download_reads.py --exp_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017
python make_pairs.py --exp_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017 -o pairs.csv
python qbic_selex_train.py --batch pairs.csv --out_dir results
python -m selection.within_experiment --results results --exp_id ETV4_eDBD_TTTGCC40NTGA_KS_yin2017
python -m selection.select_models within --results results
qbic-selex predict -v variants.csv -m results/models/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.weights.qbic \
    --compute-stats -c results/models/ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer.cov_8192.npy -o predictions.csv
```

## Training on your own reads

The training code does not depend on Table S1; give it an early-cycle and a late-cycle read file
(FASTQ, FASTA, one read per line, or a CSV with a `sequence` column; `.gz` is fine):

```bash
python qbic_selex_train.py --early_reads cycle0.fastq.gz --late_reads cycle3.fastq.gz --name MYTF_0_3
python qbic_selex_train.py --early_reads c0.txt --late_reads c2.txt --name MYTF_0_2 --trim 20
```

Reads with any character other than A/C/G/T are dropped; all reads must have the same length
(use `--trim N` to cut them to their first N bases). The outputs are described [below](#outputs).

To use your own universal bias model, train it on the reads of a TF-free control experiment, then
pass it with `--bias_model`:

```bash
python qbic_train_bias_model.py --early_reads ctrl_c0.fastq.gz --late_reads ctrl_c2.fastq.gz \
    --name MYCTRL_0_2 --out_dir bias_candidates
python qbic_selex_train.py --early_reads cycle0.fastq.gz --late_reads cycle3.fastq.gz --name MYTF_0_3 \
    --bias_model bias_candidates/MYCTRL_0_2_CNN11K128B512F_L2_DTrue.pt
```

`qbic_train_bias_model.py` runs a 108-configuration grid search with 3-fold cross-validation, then
trains the best configuration on all reads (a GPU is recommended). It writes the model as
`<name>_CNN<kernel>K<batch>B<filters>F_L<layers>_D<dropout>.pt` and the grid results as
`<name>_hyperparameter_results.csv`; the file name in the example above depends on the configuration that wins. Any checkpoint in this format works with `--bias_model`.

## Reproducing the paper's models at scale

All experiments come from Table S1 (`data/TableS1_metadata_experiments.csv`: 19,027 cycle files,
6,385 experiment IDs including the initial libraries). The full read set is about 310 GB.

```bash
# 1. reads (ENA and Zenodo; no SRA toolkit needed)
python download_reads.py --all                       # or --exp_id / --tf / --source / --model_list
# 2. the cycle pairs, one row per pair
python make_pairs.py --all -o pairs.csv
# 3. train: residual CNN + 6- and 7-mer OLS, one Slurm task per pair
sbatch --array=1-$(($(wc -l < pairs.csv) - 1))%50 workflow/train_pairs.sbatch pairs.csv results
# 4. within-experiment validation, then the top 2 models per experiment (Table S2)
tail -n +2 pairs.csv | cut -d, -f1 | uniq > exps.txt
sbatch --array=1-$(wc -l < exps.txt)%50 workflow/validate.sbatch exps.txt results
python -m selection.select_models within --results results
# 5. cross-experiment validation, then the best model per TF (Table S3)
python -m selection.cross_experiment --results results
python -m selection.select_models cross --results results
```

`results/selection/` then holds `within_experiment_top2.csv` (Table S2 columns) and
`primary_models.csv` (Table S3 columns).
To retrain only the pairs behind a released collection, list its model IDs:
`python make_pairs.py --model_list primary_997.txt -o pairs.csv` (and
`python download_reads.py --model_list primary_997.txt`).

## Randomness and reproducibility

Every random step uses the config `seed` (42):

| Step | How |
|---|---|
| CNN initial weights, shuffling | PyTorch (CPU and CUDA), seeded for each pair; cuDNN in deterministic mode |
| 1,800,000 / 450,000 / 500,000 read samples | `DataFrame.sample(random_state=seed)` |
| Bias-model cross-validation folds | `KFold(shuffle=True, random_state=42)` |

Each cycle pair is trained in its own run, seeded afresh, so a pair's model does not depend on
which other pairs are trained with it. Results are identical from run to run on the same hardware
and software; across GPU models, CUDA or library versions, floating-point differences change the
trained models slightly.

## Outputs

Under `--out_dir` (default `results/`), with `<pair>` = `<exp_id>_<early>_<late>`:

| File | Content |
|---|---|
| `cnn_scores/<pair>.csv` | `sequence,score`: f_QBiC for each unique read (the OLS input) |
| `cnn_scores/<pair>.alternate.csv` | the other score (`--save_alternate`; for corrected-vs-uncorrected comparisons) |
| `cnn_scores/<pair>.json` | α, selected score, read counts, settings, seed, run mode, versions |
| `cnn_models/<pair>.pt` | residual-model checkpoint (`--save_cnn`) |
| `models/<pair>_<k>mer.weights.qbic` | k-mer weights: pickled dict canonical k-mer → float32 (qbic-selex format) |
| `models/<pair>_<k>mer.cov_<n>.npy` | covariance of the weights: upper triangle, float32 (qbic-selex format) |
| `failed_pairs.csv` | pairs that failed, with the error (the other pairs still run) |
| `validation/{within,cross}/<model>.csv`, `selection/*.csv` | validation results and selected models |

## Configuration

All settings are in `config/config.yaml`, with the paper's values and the Methods sentence each
one implements. Use `--config my.yaml` for a modified copy; `--seed`, `--kmer`, `--bias_model`,
`--device` and `--ols_device` override single values.

## Hardware

Training one ETV4-sized pair (~450,000 reads of 40 bp) takes about 3 minutes on an NVIDIA RTX
A6000. A 7-mer OLS fit on 450,000 reads needs about 30 GB of GPU memory with CuPy; on smaller GPUs
use `--ols_device cpu` (about 32 GB of RAM). Reading all reads of a large cycle needs several GB of RAM.

## Citation

Please cite the QBiC-SELEX paper (citation to be added on publication). Models:
Li S., Gordân R. *QBiC-SELEX model weights and covariance matrices* (v2.0). Zenodo.
https://doi.org/10.5281/zenodo.23482275

## License

MIT. Questions: Shengyu Li (shengyu.li@duke.edu), or open a GitHub issue.
