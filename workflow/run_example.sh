#!/bin/bash
# End-to-end example: rebuild the released ETV4 model from raw reads.
#
#   pip install qbic-selex                 # for the comparison and prediction steps
#   bash workflow/run_example.sh           # run from the repository root; a GPU is recommended
#
# 1. downloads the 5 cycle files of ETV4_eDBD_TTTGCC40NTGA_KS_yin2017 (~45 MB, ENA)
# 2. trains all 9 cycle pairs (residual CNN, then 6- and 7-mer OLS)
# 3. within-experiment validation and the top 2 models (compare with Table S2)
# 4. compares the 0_4 7-mer model with the released one shipped with qbic-selex
# 5. predicts variant effects with the retrained model (qbic-selex predict)
set -euo pipefail
EXP=ETV4_eDBD_TTTGCC40NTGA_KS_yin2017
MODEL=${EXP}_0_4_7mer
OUT=${OUT:-results_example}
mkdir -p "$OUT"

python download_reads.py --exp_id "$EXP"
python make_pairs.py --exp_id "$EXP" -o "$OUT/pairs.csv"
python qbic_selex_train.py --batch "$OUT/pairs.csv" --out_dir "$OUT"
python -m selection.within_experiment --results "$OUT" --exp_id "$EXP"
python -m selection.select_models within --results "$OUT"
cat "$OUT/selection/within_experiment_top2.csv"

EXAMPLES=$(python -c "import pathlib, qbic_selex; print(pathlib.Path(qbic_selex.__file__).parent / 'examples')" \
           2> /dev/null) || { echo "pip install qbic-selex for steps 4-5"; exit 0; }
python workflow/compare_models.py "$OUT/models/$MODEL.weights.qbic" "$EXAMPLES/example_models/$MODEL.weights.qbic" \
    --sequences "$EXAMPLES/example_input_sequences.csv"
qbic-selex predict -v "$EXAMPLES/example_input_sequences.csv" -m "$OUT/models/$MODEL.weights.qbic" \
    --compute-stats -c "$OUT/models/$MODEL.cov_8192.npy" -o "$OUT/example_predictions.csv"
head -5 "$OUT/example_predictions.csv"
