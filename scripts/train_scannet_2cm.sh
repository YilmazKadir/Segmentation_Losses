#!/bin/bash
set -x
set -e
set -o pipefail

export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-ScannetVoxelization2cmDataset}
export MODEL=${MODEL:-Res16UNet34C}
export OPTIMIZER=${OPTIMIZER:-AdamW}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-3}
export SCHEDULER=${SCHEDULER:-OneCycleLR}
export MAX_ITER=${MAX_ITER:-120000}
export WEIGHTS=${WEIGHTS:-partition8_4096_100k.pth}
export RESUME=${RESUME:-outputs/ScannetVoxelization2cmDataset/Res16UNet34C/AdamW-l1e-2-b3-OneCycleLR-i120000-/2022-11-19_18-19-50}
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
    --train_limit_numpoints 1100000 \
    --merge True \
    --weights $WEIGHTS \
    --conv1_kernel_size 3 \
    --resume $RESUME \
    $3 2>&1 | tee -a "$LOG"
