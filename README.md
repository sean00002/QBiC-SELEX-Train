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

| Paper (Methods / Supplementary Methods) | Code |
|---|---|
| Paired-cycle design: which cycles are paired | `qbic_data/manifest.py` (`experiment_pairs`), `make_pairs.py` |
| HT-SELEX data and preprocessing (Table S1) | `qbic_data/`, `download_reads.py`, `data/TableS1_metadata_experiments.csv` |
| Experiment eligibility (≥ 2 cycles, > 10,000 reads) | `summarize_reads.py` |
| Universal bias model (architecture, use) | `qbic_train/networks.py` (`UpdatedCNN`), `bias_model/` |
| Bias-model parameter optimization and selection | `qbic_train_bias_model.py`, `qbic_train/bias_model.py`, `selection/bias_model_matrix.py` |
| TF-specific residual model and the score f_QBiC | `qbic_train/networks.py` (`ResidualModel`), `qbic_train/residual.py` |
| Interpretable k-mer OLS model, covariance for statistics | `qbic_train/kmers.py`, `qbic_train/ols.py` |
| Within- and cross-experiment validation, model curation (Tables S2, S3) | `selection/`, `qbic_train/validation.py` |
| Hyperparameters | `config/config.yaml` |
| One command per step | `qbic_selex_train.py`, `workflow/` (example and Slurm scripts) |

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
Python 3.11, PyTorch 2.14, NumPy 2.4, pandas 3.0 and CuPy 14.2 (see [Verification](#verification)).

## Quick start: rebuild a released model from raw reads

```bash
bash workflow/run_example.sh
```

This downloads the five cycles of the ETV4 experiment `ETV4_eDBD_TTTGCC40NTGA_KS_yin2017`
(~45 MB from ENA), trains all nine cycle pairs (one pair per run, as for the published models),
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

## Reproducing the paper's models at scale

All experiments come from Table S1 (`data/TableS1_metadata_experiments.csv`: 19,027 cycle files,
6,385 experiment IDs including the initial libraries). The full read set is about 310 GB.

```bash
# 1. reads (ENA and Zenodo; no SRA toolkit needed)
python download_reads.py --all                       # or --exp_id / --tf / --source / --model_list
# 2. every cycle pair (37,111 pairs over all Table S1 experiments)
python make_pairs.py --all -o pairs.csv              # one row per pair
python make_pairs.py --all --per_experiment -o experiments.csv   # or one row per experiment
# 3. train: residual CNN + 6- and 7-mer OLS, one Slurm task per row
sbatch --array=1-37111%50 workflow/train_pairs.sbatch pairs.csv results
# 4. within-experiment validation, then the top 2 models per experiment (Table S2)
sbatch --array=1-N%50 workflow/validate.sbatch exps.txt results
python -m selection.select_models within --results results
# 5. cross-experiment validation, then the best model per TF (Table S3)
python -m selection.cross_experiment --results results
python -m selection.select_models cross --results results
```

The paper's collection came from the 4,362 experiments with at least two cycles and more than
10,000 reads per cycle (36,938 residual models, 73,876 OLS models); `summarize_reads.py` reports
reads, unique reads and read length per cycle file and flags experiments that meet this rule. `results/selection/` then holds
`within_experiment_top2.csv` (Table S2 columns) and `primary_models.csv` (Table S3 columns).
To retrain only the pairs behind a released collection, list its model IDs:
`python make_pairs.py --model_list primary_997.txt -o pairs.csv` (and
`python download_reads.py --model_list primary_997.txt`).

## How the code follows the Methods

**Cycle pairs.** For each experiment, the early cycle is 0, 1 or 2 and the late cycle is any later
cycle. Cycle 0 is the initial library with the same barcode and source (`record_class = cycle-zero`
in Table S1). For Bhimsaria et al. (2023), the bead-only selections 0(1), 0(2) and 0(3) are extra
early cycles. Pairs are named `<exp_id>_<early>_<late>` and models `<pair>_<k>mer`, as in Tables S2–S6.

**Reads.** Reads containing N are removed; duplicates are kept for CNN training. Bhimsaria et al.
reads are cut to their first 20 bp; Rodríguez-Martínez et al. (2017) files already hold the 20 bp
random region.

**Residual model.** One convolution (128 filters, kernel 9), ReLU, global max pooling and a linear
output give f_RES(x); with the frozen universal bias model f_BIAS(x) and α = tanh(a), the
classifier is f_SELEX(x) = α·f_BIAS(x) + f_RES(x). Training: weighted binary cross-entropy (late
cycle positive), Adam (learning rate 10⁻³, ε 10⁻⁴, AMSGrad), gradient clipping at 1.0, 7 epochs,
batch size 64, seed 42. The bias model's weights receive no gradient; as in the original code, its
batch-normalization layers are in training mode while the residual model trains, so their running
statistics adapt to the paired-cycle data.

**TF binding score.** Every read is scored; duplicates are collapsed to one row and at most
1,800,000 unique reads are kept (sampled with seed 42). The OLS input is
f_QBiC = f_RES if α ≥ 0, else f_SELEX (the two are identical at α = 0). `cnn_scores/<pair>.json`
records α and which score was used.

**k-mer OLS.** Each k-mer and its reverse complement share one feature (d = 2,080 for 6-mers,
8,192 for 7-mers). 6-mer models use all scored reads; 7-mer models a random 450,000 (seed 42).
β = (XᵀX)⁻¹Xᵀy and Σ = σ̂²(XᵀX)⁻¹ with σ̂² = RSS/(N − d), in float32, on the GPU with CuPy when
available. QBiC-SELEX uses β to predict variant effects and Σ for their z-scores and P-values.

**Validation and curation.** Within an experiment, each OLS model predicts the CNN scores of every
cycle pair of that experiment, its own pair included, on a random 500,000 unique reads per pair
(without replacement, seed 42); the model's score is the mean Pearson r, and the two best models of
each experiment are kept (Table S2). Each of these then predicts the CNN scores of the top-2 pairs
of every experiment of the same TF, its own experiment included; the model with the highest mean
r is the TF's primary model (Table S3). Ties keep the model whose ID sorts first.

**Universal bias model.** `bias_model/beadOnlyControl_NA_TG40NCATATG_AATBA_jolma2025_0_2_CNN11K128B512F_L2_DTrue.pt`
is the model used for every TF model in the paper (sha256 `76e612e0…5e69c184e`, checked at load
time): two convolutional layers of 512 filters (kernel 11) with batch normalization and ReLU,
global max pooling, dropout 0.2, a 64-unit hidden layer; trained on cycles 0 vs 2 of the bead-only
control of Jolma et al. (2025), batch size 128, 10 epochs. It was chosen as follows, and the code
for both steps is included:

```bash
# one candidate per TF-free cycle pair: 108-configuration grid search, 3-fold CV, final fit
python qbic_train_bias_model.py --exp_id beadOnlyControl_NA_TG40NCATATG_AATBA_jolma2025 --early 0 --late 2
# every candidate on every TF-free pair (Supplementary Fig. 2); the best mean AUROC wins
python -m selection.bias_model_matrix --models bias_candidates/*.pt --pairs <pair IDs>
```

The original grid search did not fix PyTorch's seed, so new candidates are reproducible
(`seed` in the config) but not bit-identical to the paper's.

## Randomness and reproducibility

Every random step uses the config `seed` (42):

| Step | How |
|---|---|
| CNN initial weights, shuffling, bias-model dropout | PyTorch (CPU and CUDA), seeded once per run; cuDNN in deterministic mode |
| 1,800,000 / 450,000 / 500,000 read samples | `DataFrame.sample(random_state=seed)` |
| Bias-model cross-validation folds | `KFold(shuffle=True, random_state=42)` |

A run seeds once and then trains its pairs in order. `--pair_id`, `--model_id`,
`--exp_id X --early a --late b` and each row of `--batch` train one pair in a fresh run; this is how
the published models were trained (see [Verification](#verification)). `--exp_id X` alone trains
all pairs of an experiment in one run, continuing one random stream (and the bias model's
batch-normalization statistics) from pair to pair, so its models differ slightly; the original
scripts offered both modes. `--model_id … --experiment_stream` trains an experiment's pairs up to
and including that one in one stream. Results are identical from run to run on the same hardware
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

## Verification

The rewritten code was checked against the original training scripts that produced the published
models (run unmodified), on the ETV4 experiment `ETV4_eDBD_TTTGCC40NTGA_KS_yin2017` (five cycles,
nine pairs), on an NVIDIA RTX A6000 with PyTorch 2.14 (CUDA 13) and CuPy 14.2:

| Check | Result |
|---|---|
| Cycle files and pair order from Table S1 vs the original lookup code (all 4,421 experiments, 37,111 pairs) | identical |
| Whole-experiment run: score files, 6- and 7-mer weights and covariances of all nine pairs | byte-identical |
| Single-pair run (`0_4`): the same files | byte-identical |
| Within-experiment validation (162 correlations) | identical (to 1e-16) |
| The original scripts, run twice | identical (deterministic) |
| Retrained `0_4_7mer` vs the released model | single-pair run: weights r = 0.992, predicted effects r = 0.987; whole-experiment run: 0.949 / 0.979 |
| Top 2 models of the experiment vs Table S2 | the same two models, `0_3_7mer` and `0_4_7mer`: r = 0.9294 and 0.9297 (single-pair runs), 0.9298 and 0.9287 (whole-experiment run); Table S2: 0.9302 and 0.9299 |
| Bias model on its own training reads (bead-only control, cycles 0 vs 2) | AUROC 0.825 (stored final training AUROC, with dropout: 0.802) |
| ETV4 primary model vs Table S3 (the 16 top-2 models of all 8 ETV4 experiments, retrained one pair per run) | the same model, `ETV4_eDBD_TTTGCC40NTGA_KS_yin2017_0_4_7mer`: cross-experiment r = 0.9230 (Table S3: 0.9237); next best 0.9211 |
| A `--batch` row vs the same pair trained with `--pair_id` | byte-identical |

The published models were trained with older PyTorch and CUDA versions on other GPUs, so
retrained models agree closely but not bit for bit: small floating-point differences change the
CNN's training path. Single-pair runs reproduce the released model much more closely than
whole-experiment runs, which is why single-pair runs are the default. `workflow/run_example.sh`
repeats the comparison with the released model.

## Tests

```bash
pip install pytest
pytest -q
```

The tests run on CPU in under a minute and check the pairing rule (including the Methods and
Supplementary Methods examples), read preparation, the score-selection rule, k-mer order and file
formats against `qbic-selex`, OLS on planted coefficients, seeded determinism, and the selection
logic behind Tables S2 and S3.

## Citation

Please cite the QBiC-SELEX paper (citation to be added on publication). Models:
Li S., Gordân R. *QBiC-SELEX model weights and covariance matrices* (v2.0). Zenodo.
https://doi.org/10.5281/zenodo.23482275

## License

MIT. Questions: Shengyu Li (shengyu.li@duke.edu), or open a GitHub issue.
