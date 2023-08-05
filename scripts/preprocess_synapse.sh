#!/usr/bin/env bash

TRANSUNET_DIR="/images/PublicDatasets/Transunet_synaps/project_TransUNet/data/Synapse/"

cp -r $TRANSUNET_DIR ./data

mkdir ./data/Synapse/validation_npz

python lib/datasets/preprocess/preprocess_synapse.py