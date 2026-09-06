#!/usr/bin/env bash
set -euo pipefail

WEIGHT_DIR=${1:-weights}
mkdir -p "$WEIGHT_DIR"

# DiffBIR v2.1 / SD2.1-zsnr / face SwinIR weights for identity-stage training.
wget -nc -O "$WEIGHT_DIR/sd2.1-base-zsnr-laionaes5.ckpt" \
  https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt
wget -nc -O "$WEIGHT_DIR/DiffBIR_v2.1.pt" \
  https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt
wget -nc -O "$WEIGHT_DIR/face_swinir_v1.ckpt" \
  https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt

# Optional ArcFace ONNX identity encoder from InsightFace buffalo_l.
wget -nc -O "$WEIGHT_DIR/buffalo_l.zip" \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -n "$WEIGHT_DIR/buffalo_l.zip" -d "$WEIGHT_DIR"

echo "weights downloaded to $WEIGHT_DIR"
echo "identity_encoder_path example: $WEIGHT_DIR/buffalo_l/w600k_r50.onnx"
