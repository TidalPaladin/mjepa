#!/bin/bash
DEVICE=cuda:0
OFFSET=0
NUM_COMPONENTS=6
SIZE="64 64"
OUTPUT=./analysis/cifar10/
DATA=/mnt/data/cifar10/cifar-10-batches-py

CONFIG=config/cifar10/mjepa/vit-small.yaml
BACKBONE=output/cifar10/2025-06-29_23-44-56_mjepa-vit-s-fourier-linprobe

#pdm run scripts/cifar10_visualize.py \
#    $CONFIG \
#    $BACKBONE/backbone.safetensors \
#    $DATA \
#    $OUTPUT/pca.png \
#    -d $DEVICE -o $OFFSET -n $NUM_COMPONENTS -dt bf16 -s $SIZE
#
#pdm run scripts/cifar10_norms.py \
#    $CONFIG \
#    $BACKBONE/backbone.safetensors \
#    $DATA \
#    $OUTPUT/norms.png \
#    -d $DEVICE
#
#pdm run scripts/position_visualize.py \
#    $CONFIG \
#    $BACKBONE/backbone.safetensors \
#    $OUTPUT/positions.png \
#    -d $DEVICE -dt bf16 

pdm run scripts/runtime_visualize.py \
    $CONFIG \
    $BACKBONE/backbone.safetensors \
    $OUTPUT/runtime.png \
    -d $DEVICE -dt bf16 --scales 0.25 0.5 1.0 2.0 4.0 -b 8
