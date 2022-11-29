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
export OPTIMIZER=${OPTIMIZER:-AdamW}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-4}
export SCHEDULER=${SCHEDULER:-OneCycleLR}
export MAX_ITER=${MAX_ITER:-120000}
# export WEIGHTS=${WEIGHTS:-/home/yilmaz/my_git_repos/a40/Minkowski_MAE/outputs/KITTIVoxelizationDataset/Res16UNet25C/AdamW-l1e-2-b4-OneCycleLR-i80000-/2022-11-01_13-15-38/checkpoint_Res16UNet25Cbest_val.pth}
export WEIGHTS=${WEIGHTS:-None}
# export RESUME=${RESUME:-outputs/KITTIVoxelizationDataset/Res16UNet34C/AdamW-l1e-2-b4-OneCycleLR-i120000-/2022-11-03_00-55-53}
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
    --merge True \
    --val_freq 4000 \
    --normalize_color False \
    $3 2>&1 | tee -a "$LOG"
