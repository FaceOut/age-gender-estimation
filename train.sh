#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

# DATA_DIR='/content/drive/MyDrive/datasets/cumulatice_aginig'

DATA_DIR='/content/drive/MyDrive/datasets/aging_mxnet_fixed'

NETWORK=m1
DATASET=ffhq
MODELDIR='./models_aug_comp_color'
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model-$NETWORK-$DATASET"
LOGFILE="$MODELDIR/log_m1_aging"
PRETRAINED="/content/drive/MyDrive/repo/age-gender-estimation/model/m1/model,0"
# PRETRAINED="/content/drive/MyDrive/repo/age-gender-estimation/models_full/model-m1-ffhq_imdb,49"
# --pretrained "$PRETRAINED"
CUDA_VISIBLE_DEVICES='0' python -u train.py --data-dir $DATA_DIR --prefix $PREFIX --pretrained "$PRETRAINED" --network $NETWORK  --per-batch-size 128  --lr 0.01 --lr-steps '10000' --ckpt 2 --verbose 500 --color 2 --multiplier 0.25 > "$LOGFILE" 2>&1 &