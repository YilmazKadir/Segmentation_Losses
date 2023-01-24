#!/bin/bash
set -x
set -e
set -o pipefail

export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-CityScapesDataset}
export MODEL=${MODEL:-UNet}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-2}
export MAX_EPOCHS=${MAX_EPOCHS:-50}
export WEIGHTS=${WEIGHTS:-None}
# export RESUME=${RESUME:-./outputs/StanfordArea5Dataset/Res16UNet34C/AdamW-l1e-2-b4-OneCycleLR-i25000-/2022-09-29_15-50-45/}

export LOG_DIR=./outputs/$DATASET/$MODEL/${OPTIMIZER}-l$LR-b$BATCH_SIZE-$SCHEDULER-i$MAX_EPOCHS/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --weights $WEIGHTS \
    --val_freq 1000 \
    $3 2>&1 | tee -a "$LOG"