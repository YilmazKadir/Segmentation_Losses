#!/bin/bash
set -x
set -e
set -o pipefail

export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-StanfordArea5Dataset}
export MODEL=${MODEL:-Res16UNet34C}
export OPTIMIZER=${OPTIMIZER:-AdamW}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-4}
export SCHEDULER=${SCHEDULER:-OneCycleLR}
export MAX_ITER=${MAX_ITER:-15000}
# export WEIGHTS=${WEIGHTS:-/home/yilmaz/my_git_repos/a40/Minkowski_MAE/outputs/ScannetVoxelizationDataset/Res16UNet25C/AdamW-l1e-2-b8-OneCycleLR-i40000-/2022-10-15_02-32-12/checkpoint_Res16UNet25Cbest_val.pth}
export WEIGHTS=${WEIGHTS:-None}
# export RESUME=${RESUME:-./outputs/StanfordArea5Dataset/Res16UNet34C/AdamW-l1e-2-b4-OneCycleLR-i25000-/2022-09-29_15-50-45/}

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
    --merge True \
    --weights $WEIGHTS \
    --save_freq 600 \
    --val_freq 600 \
    $3 2>&1 | tee -a "$LOG"