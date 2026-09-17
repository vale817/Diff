#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/DiffBIR_identity
PY=/root/miniconda3/envs/diffbir/bin/python
INPUT=/root/autodl-tmp/eval_inputs/ffhq_lq_test1000_clean
GT=/root/autodl-tmp/eval_gt/ffhq_gt_test1000_clean
POS_OUT=/root/autodl-tmp/results/positive_only1000_train_ffhq_test1000
CTL_OUT=/root/autodl-tmp/results/diffusion_control1000_train_ffhq_test1000
POS_EVAL=/root/autodl-tmp/experiments/positive_only_eval_test1000
CTL_EVAL=/root/autodl-tmp/experiments/diffusion_control_eval_test1000
STATUS=/root/autodl-tmp/experiments/ffhq_test1000_continuation.status

export LD_LIBRARY_PATH=/root/miniconda3/envs/diffbir/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/root/miniconda3/envs/diffbir/lib/python3.10/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/diffbir/lib/python3.10/site-packages/nvidia/cufft/lib:/root/miniconda3/envs/diffbir/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

count_png() {
  if [[ ! -d "$1" ]]; then
    echo 0
    return 0
  fi
  find "$1" -maxdepth 1 -type f -name '*.png' | wc -l
}

printf 'waiting_positive\n' > "$STATUS"
while pgrep -f 'inference.py.*positive_only1000_train_ffhq_test1000' >/dev/null; do
  sleep 30
done

pos_count=$(count_png "$POS_OUT")
if [[ "$pos_count" -ne 1000 ]]; then
  printf 'failed_positive_count=%s\n' "$pos_count" > "$STATUS"
  exit 1
fi

ctl_count=$(count_png "$CTL_OUT")
if [[ "$ctl_count" -lt 1000 ]]; then
  printf 'running_control count=%s\n' "$ctl_count" > "$STATUS"
  cd "$REPO"
  "$PY" inference.py \
    --task face --version v2.1 --upscale 1 \
    --input "$INPUT" --output "$CTL_OUT" \
    --identity_ckpt /root/autodl-tmp/experiments/diffusion_only_control_1000/checkpoints/0001000.pt \
    --identity_encoder /root/autodl-tmp/DiffBIR_identity/weights/buffalo_l/w600k_r50.onnx \
    --identity_scale 1 --sampler spaced --steps 50 --cfg_scale 4.0 \
    --captioner none --pos_prompt '' \
    --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
    --seed 231 --precision fp16 --batch_size 1 --n_samples 1 \
    > /root/autodl-tmp/experiments/diffusion_control1000_ffhq_test1000_infer.log 2>&1
fi

ctl_count=$(count_png "$CTL_OUT")
if [[ "$ctl_count" -ne 1000 ]]; then
  printf 'failed_control_count=%s\n' "$ctl_count" > "$STATUS"
  exit 1
fi

mkdir -p "$POS_EVAL" "$CTL_EVAL"
printf 'evaluating_quality\n' > "$STATUS"
"$PY" /root/autodl-tmp/DiffBIR/tools/calc_ref_metrics.py \
  --gt_dir "$GT" --pred_dir "$POS_OUT" --method positive_only1000_train \
  --out_dir "$POS_EVAL" > "$POS_EVAL/quality.log" 2>&1
"$PY" /root/autodl-tmp/DiffBIR/tools/calc_ref_metrics.py \
  --gt_dir "$GT" --pred_dir "$CTL_OUT" --method diffusion_control1000_train \
  --out_dir "$CTL_EVAL" > "$CTL_EVAL/quality.log" 2>&1

printf 'evaluating_identity\n' > "$STATUS"
"$PY" /root/autodl-tmp/DiffBIR/tools/eval_arcface.py \
  --arcface_onnx /root/autodl-tmp/DiffBIR_identity/weights/buffalo_l/w600k_r50.onnx \
  --restored_dir "$POS_OUT" --reference_dir "$GT" \
  --output_csv "$POS_EVAL/arcface.csv" --batch_size 100 --device cuda \
  > "$POS_EVAL/arcface.log" 2>&1
"$PY" /root/autodl-tmp/DiffBIR/tools/eval_adaface.py \
  --adaface_encoder /root/autodl-tmp/DiffBIR/weights/adaface_ir50_ms1mv2_torchscript.pt \
  --restored_dir "$POS_OUT" --reference_dir "$GT" \
  --output_csv "$POS_EVAL/adaface.csv" --batch_size 100 --device cuda \
  > "$POS_EVAL/adaface.log" 2>&1
"$PY" /root/autodl-tmp/DiffBIR/tools/eval_arcface.py \
  --arcface_onnx /root/autodl-tmp/DiffBIR_identity/weights/buffalo_l/w600k_r50.onnx \
  --restored_dir "$CTL_OUT" --reference_dir "$GT" \
  --output_csv "$CTL_EVAL/arcface.csv" --batch_size 100 --device cuda \
  > "$CTL_EVAL/arcface.log" 2>&1
"$PY" /root/autodl-tmp/DiffBIR/tools/eval_adaface.py \
  --adaface_encoder /root/autodl-tmp/DiffBIR/weights/adaface_ir50_ms1mv2_torchscript.pt \
  --restored_dir "$CTL_OUT" --reference_dir "$GT" \
  --output_csv "$CTL_EVAL/adaface.csv" --batch_size 100 --device cuda \
  > "$CTL_EVAL/adaface.log" 2>&1

printf 'complete positive=1000 control=1000\n' > "$STATUS"
