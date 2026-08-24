# Changelog

## Unreleased

### Fixed

- Corrected the alpha-dependent selection of CNN scores used for downstream
  OLS training. Earlier versions of the modular public pipeline returned the
  outputs from `ResidualTrainer._generate_predictions()` in the opposite order
  from the caller's variable names, causing the alternate score to be written
  to `residual_model_output/corrected/`.
- The selected behavior is now explicit: for `alpha >= 0`, QBiC uses the
  TF-specific residual output `f_RES`; for `alpha < 0`, QBiC uses the complete
  output `f_SELEX = f_RES + alpha * f_BIAS`.
- Added score-semantics metadata alongside residual outputs and in optional
  residual-model checkpoints.

This issue affects models newly trained with the affected modular public
pipeline. It does not change the QBiC-SELEX architecture or the production
selection rule used by the original training scripts. Affected modular outputs
should be regenerated rather than renamed unless their alpha value and score
provenance are known.
