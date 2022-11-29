#!/bin/bash
set -x
set -e
set -o pipefail

export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-ScannetVoxelizationDataset}
export MODEL=${MODEL:-Res16UNet34C}
export OPTIMIZER=${OPTIMIZER:-AdamW}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-4}
export SCHEDULER=${SCHEDULER:-OneCycleLR}
export MAX_ITER=${MAX_ITER:-90000}
export WEIGHTS=${WEIGHTS:-/home/yilmaz/my_git_repos/a40/Minkowski_MAE/outputs/ScannetVoxelizationDataset/Res16UNet25C/AdamW-l1e-2-b8-OneCycleLR-i50000-/2022-11-02_14-13-29/checkpoint_Res16UNet25Cbest_val.pth}
# export RESUME=${RESUME:-outputs/ScannetVoxelizationDataset/Res16UNet34C/AdamW-l1e-2-b4-OneCycleLR-i90000-/2022-10-05_23-56-06}
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
    $3 2>&1 | tee -a "$LOG"
