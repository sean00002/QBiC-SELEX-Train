# Changelog

## 2026-10 — cleaned-up release for the paper

Rewritten from the original training scripts into documented modules, with the same seeds,
order of random draws and formulas, and checked against them on real data (README, Verification).

### Added
- `qbic_data/`, `download_reads.py`, `make_pairs.py`: the paper's data, seeded by Table S1
  (cycle files, the paired-cycle rule including the Bhimsaria et al. bead-only cycles,
  source-specific preprocessing) and downloads from ENA/Zenodo without the SRA toolkit.
- `selection/`: within- and cross-experiment validation and model curation (Tables S2 and S3),
  and the bias-candidate AUROC matrix (Supplementary Fig. 2).
- Whole-experiment runs (`--exp_id`), single pairs (`--pair_id`, `--model_id`), batches with
  `--row` for Slurm arrays, `--skip_existing`, and both 6- and 7-mer models by default.
- `workflow/`: an end-to-end example that rebuilds a released model, and Slurm scripts.
- pytest suite and GitHub Actions CI; MIT license.

### Changed
- Model files are named `<pair>_<k>mer.weights.qbic` and `<pair>_<k>mer.cov_<n>.npy`, as
  `qbic-selex` expects (previously `<pair>_<k>mer.qbic`).
- Output layout: `cnn_scores/` (selected score f_QBiC, optional alternate score, JSON metadata),
  `cnn_models/`, `models/`.
- All hyperparameters are read from `config/config.yaml` (values unchanged).
- OLS runs on CPU (NumPy) when CuPy is not available; cuDF/cuML are no longer needed.

### Fixed
- Bhimsaria et al. pairs (cycles such as `0(2)`) can be trained.
- The bundled bias model loads with PyTorch ≥ 2.6.
- A failing pair is logged in `failed_pairs.csv` instead of being skipped silently, and the
  other pairs continue.
- No hard-coded paths.

### Removed
- `modules/`, `util_scripts/`, `download_raw/` (the download list duplicated Table S1),
  `test_pipeline.py`, and committed `__pycache__` files.

## Earlier (2026-08)

- Corrected the alpha-dependent selection of CNN scores in the first modular version of this
  repository (its outputs were returned in the opposite order). Production models were made with
  the original scripts, which used the intended rule (f_RES if α ≥ 0, else f_SELEX).
