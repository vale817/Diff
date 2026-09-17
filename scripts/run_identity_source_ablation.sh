#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_identity_source_ablation.sh \
    TEST_LQ_DIR OUTPUT_ROOT IDENTITY_ENCODER STAGE1_CKPT LQ_CKPT \
    [additional inference.py arguments]

The same additional arguments are used for both runs. Example:
  bash scripts/run_identity_source_ablation.sh \
    /root/autodl-tmp/test1000/lq \
    /root/autodl-tmp/results/identity_source_ablation \
    /root/autodl-tmp/DiffBIR_identity/weights/buffalo_l/w600k_r50.onnx \
    /root/autodl-tmp/experiments/identity_stage1_5k/checkpoints/last.pt \
    /root/autodl-tmp/experiments/identity_lq_5k/checkpoints/last.pt \
    --sampler edm_dpm++_3m_sde --steps 10 --captioner none --precision fp16
EOF
  exit 2
fi

test_lq_dir=$1
output_root=$2
identity_encoder=$3
stage1_ckpt=$4
lq_ckpt=$5
shift 5

for required_path in "$test_lq_dir" "$identity_encoder" "$stage1_ckpt" "$lq_ckpt"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$output_root"

common_args=(
  --task face
  --version v2.1
  --upscale 1
  --input "$test_lq_dir"
  --seed 231
  --identity_encoder "$identity_encoder"
  --identity_scale 1.0
)

python inference.py \
  "${common_args[@]}" \
  --identity_ckpt "$stage1_ckpt" \
  --identity_source stage1 \
  --output "$output_root/stage1_source" \
  "$@" \
  2>&1 | tee "$output_root/stage1_source.log"

python inference.py \
  "${common_args[@]}" \
  --identity_ckpt "$lq_ckpt" \
  --identity_source lq \
  --output "$output_root/lq_source" \
  "$@" \
  2>&1 | tee "$output_root/lq_source.log"
