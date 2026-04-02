#!/bin/bash
# ============================================================
# MASTER RUN SCRIPT
# Run this after training is complete to get all L2-L4 results.
# Usage: bash run_all.sh
# ============================================================

set -e  # exit on error

DATA_DIR="data/DRIVE"
CKPT_DIR="checkpoints"
RESULTS="results"
MC_CKPT="$CKPT_DIR/unet_mc_dropout/best_model.pth"

echo "========================================"
echo " STEP 1: Evaluate MC Dropout model"
echo "========================================"
python evaluate.py \
    --data_dir $DATA_DIR \
    --checkpoint $MC_CKPT \
    --n_passes 30 \
    --output_dir $RESULTS/mc_dropout

echo ""
echo "========================================"
echo " STEP 2: Evaluate Deep Ensemble"
echo "========================================"
python evaluate_ensemble.py \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CKPT_DIR \
    --output_dir $RESULTS/ensemble

echo ""
echo "========================================"
echo " STEP 3: Compare all methods (L2+L3)"
echo "========================================"
python experiments/compare_methods.py \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CKPT_DIR \
    --output_dir $RESULTS/comparison \
    --n_passes 30

echo ""
echo "========================================"
echo " STEP 4: Ablation study (L4)"
echo "========================================"
python experiments/ablation.py \
    --data_dir $DATA_DIR \
    --checkpoint $MC_CKPT \
    --output_dir $RESULTS/ablation

echo ""
echo "========================================"
echo " STEP 5: Cross-dataset evaluation (L4)"
echo "========================================"
python experiments/cross_dataset.py \
    --checkpoint $MC_CKPT \
    --stare_dir data/STARE \
    --chase_dir data/CHASE_DB1 \
    --output_dir $RESULTS/cross_dataset

echo ""
echo "========================================"
echo " ALL DONE"
echo " Results in: $RESULTS/"
echo " Copy figures to paper/figures/"
echo " Fill in TODO values in paper/main.tex"
echo "========================================"
