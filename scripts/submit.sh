#/bin/bash
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export TARGET_DIR=/home/students/kyilmaz/copy/$TIME
export CODE_DIR=/home/students/kyilmaz/Segmentation_Losses
mkdir $TARGET_DIR
cp -r $CODE_DIR $TARGET_DIR
cd $TARGET_DIR/Segmentation_Losses
condor_submit initialdir=$TARGET_DIR/Segmentation_Losses remote_initialdir=$TARGET_DIR/Segmentation_Losses executable=$TARGET_DIR/Segmentation_Losses/scripts/train_synapse.sh scripts/run.tbi