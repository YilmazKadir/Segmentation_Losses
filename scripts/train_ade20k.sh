#!/bin/bash
set -x
set -e
set -o pipefail

export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-ADE20KDataset}
export MODEL=${MODEL:-UNet}
export OPTIMIZER=${OPTIMIZER:-AdamW}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-4}
export SCHEDULER=${SCHEDULER:-OneCycleLR}
export MAX_ITER=${MAX_ITER:-40000}
export WEIGHTS=${WEIGHTS:-None}
# export RESUME=${RESUME:-outputs/ScannetVoxelizationDataset/Res16UNet34C/AdamW-l1e-2-b4-OneCycleLR-i90000-/2022-10-05_23-56-06}
export LOG_DIR=/work/scratch/kyilmaz/Segmentation_Losses/outputs/$DATASET/$MODEL/-l$LR-b$BATCH_SIZE-$SCHEDULER-i$MAX_ITER-$EXPERIMENT/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

/home/students/kyilmaz/miniconda3/envs/segmentation_losses/bin/python main.py $@ \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --scheduler $SCHEDULER \
    --max_iter $MAX_ITER \
    --weights $WEIGHTS \
    --val_freq 4000 \
    2>&1 | tee -a "$LOG"