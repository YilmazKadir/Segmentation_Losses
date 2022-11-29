#!/bin/bash
set -x
set -e
set -o pipefail

export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-TemporalKITTIVoxelizationDataset}
export MODEL=${MODEL:-STRes16UNet34C}
export OPTIMIZER=${OPTIMIZER:-AdamW}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-3}
export SCHEDULER=${SCHEDULER:-OneCycleLR}
export MAX_ITER=${MAX_ITER:-20000}
export WEIGHTS=${WEIGHTS:-None}
# export RESUME=${RESUME:-outputs/TemporalKITTIVoxelizationDataset/STRes16UNet34C/AdamW-l1e-2-b3-OneCycleLR-i20000-/2022-11-20_13-36-41}
export LOG_DIR=./outputs/$DATASET/$MODEL/${OPTIMIZER}-l$LR-b$BATCH_SIZE-$SCHEDULER-i$MAX_ITER-$EXPERIMENT/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

srun python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --scheduler $SCHEDULER \
    --max_iter $MAX_ITER \
    --weights $WEIGHTS \
    --train_limit_numpoints 1200000 \
    --merge False \
    --val_freq 4000 \
    --normalize_color False \
    $3 2>&1 | tee -a "$LOG"
