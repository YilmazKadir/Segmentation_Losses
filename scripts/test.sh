#!/bin/bash
set -x
set -e
set -o pipefail

export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-KITTIVoxelizationDataset}
export MODEL=${MODEL:-Res16UNet34C}
export BATCH_SIZE=${BATCH_SIZE:-4}
export WEIGHTS=${WEIGHTS:-outputs/KITTIVoxelizationDataset/Res16UNet34C/AdamW-l1e-2-b4-OneCycleLR-i120000-/2022-11-03_00-55-53/checkpoint_Res16UNet34C.pth}
export LOG_DIR=outputs/$DATASET/$MODEL/test/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --is_train False \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --weights $WEIGHTS \
    --test_original_pc True \
    --test_phase val \
    --normalize_color False \
    $3 2>&1 | tee -a "$LOG"
