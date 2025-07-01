#!/bin/bash
DEVICE=cuda:0
OFFSET=0
NUM_COMPONENTS=6
SIZE="512 512"
OUTPUT=./analysis/chest/
CONFIG=config/chest-xray/mjepa/vit-base.yaml

BACKBONE=output/chest-xray/2025-06-29_23-36-16_mjepa-vit-b-256x256-fourier-simple

DATA=test_images

pdm run scripts/pca_visualize.py \
    $CONFIG \
    $BACKBONE/backbone.safetensors \
    ${DATA[@]} \
    $OUTPUT/pca.png \
    -d $DEVICE -o $OFFSET -n $NUM_COMPONENTS -dt bf16 -s $SIZE

pdm run scripts/plot_norms.py \
    $CONFIG \
    $BACKBONE/backbone.safetensors \
    ${DATA[@]} \
    $OUTPUT/norms.png \
    -d $DEVICE

pdm run scripts/position_visualize.py \
    $CONFIG \
    $BACKBONE/backbone.safetensors \
    $OUTPUT/positions.png \
    -d $DEVICE -dt bf16 

pdm run scripts/runtime_visualize.py \
    $CONFIG \
    $BACKBONE/backbone.safetensors \
    $OUTPUT/runtime.png \
    -d $DEVICE -dt bf16 --scales 0.25 0.5 1.0 2.0 4.0 -b 8
