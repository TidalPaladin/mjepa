#!/bin/bash
DEVICE=cuda:0
OFFSET=0
NUM_COMPONENTS=6
SIZE="512 384"
OUTPUT=./analysis/mammo/
CONFIG=config/mammo/mjepa/vit-base.yaml
BACKBONE=output/mammo/2025-06-28_14-45-22_mjepa-vit-b-256x192-fourier-simple

DATA=(
    "/mnt/data/vindr/highres/004426a40da27ef22a866538b772ac44/265596e3534efced063b4e656b7bd64f.tiff" \
    "/mnt/data/vindr/highres/004426a40da27ef22a866538b772ac44/7197f7e6f5ca3b9821d0460dcdc975d3.tiff" \
    "/mnt/data/vindr/highres/004426a40da27ef22a866538b772ac44/78b8a2f4b10790d6b2a11e081eaf209a.tiff" \
    "/mnt/data/vindr/highres/004426a40da27ef22a866538b772ac44/d3ae4a0aa59a608a2823c029e988985f.tiff"
)

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
    $OUTPUT/runtime.png \
    -d $DEVICE -dt bf16 --scales 0.25 0.5 1.0 2.0 4.0 -b 4
