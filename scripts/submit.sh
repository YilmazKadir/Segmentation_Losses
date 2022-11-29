#/bin/bash
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export TARGET_DIR=/work/scratch/kyilmaz/copy/$TIME
CODE_DIR=/work/scratch/kyilmaz/Segmentation_Losses
mkdir $TARGET_DIR
cp -r $CODE_DIR $TARGET_DIR
cd $TARGET_DIR/Segmentation_Losses
condor_submit initialdir=$TARGET_DIR/Segmentation_Losses remote_initialdir=$TARGET_DIR/Segmentation_Losses executable=$TARGET_DIR/Segmentation_Losses/scripts/train_ade20k.sh scripts/run.tbi