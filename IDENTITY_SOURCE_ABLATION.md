# Identity-source ablation

This experiment compares two identity inputs while keeping the original
SwinIR-to-ControlNet restoration path unchanged.

| Variant | Training identity input | Inference identity input |
| --- | --- | --- |
| Stage-1 source | SwinIR coarse restoration | SwinIR coarse restoration |
| LQ source | LQ tensor supplied to SwinIR | LQ tensor supplied to SwinIR |

## Controlled variables

Use the same base DiffBIR, SwinIR, ArcFace encoder, train list, degradation,
seed, optimizer, learning rate, training steps, sampling settings, test LQ
images, and metric implementation. Start both identity adapters from a fresh
initialization; do not resume the LQ run from the stage-1-source checkpoint.

The existing stage-1-source checkpoint can be reused if it was trained with
the settings in `configs/train/train_identity_attention_stage1_5k.yaml`.

## Train the LQ-source adapter

Check the paths in `configs/train/train_identity_attention_lq_5k.yaml`, then
run:

```bash
mkdir -p /root/autodl-tmp/experiments/identity_lq_5k

python train_identity_attention_stage1.py \
  --config configs/train/train_identity_attention_lq_5k.yaml \
  --seed 231 \
  --mixed_precision fp16 \
  --gradient_accumulation_steps 1 \
  2>&1 | tee /root/autodl-tmp/experiments/identity_lq_5k/train.log
```

The checkpoint should be written to:

```text
/root/autodl-tmp/experiments/identity_lq_5k/checkpoints/last.pt
```

## Run the paired test1000 inference

Use the same inference arguments as the main ID-DiffBIR test1000 run. The
wrapper below sends every extra argument to both variants:

```bash
bash scripts/run_identity_source_ablation.sh \
  TEST1000_LQ_DIR \
  /root/autodl-tmp/results/identity_source_ablation \
  /root/autodl-tmp/DiffBIR_identity/weights/buffalo_l/w600k_r50.onnx \
  /root/autodl-tmp/experiments/identity_stage1_5k/checkpoints/last.pt \
  /root/autodl-tmp/experiments/identity_lq_5k/checkpoints/last.pt \
  --sampler edm_dpm++_3m_sde \
  --steps 10 \
  --captioner none \
  --precision fp16
```

Replace the final inference arguments with the exact settings used for the
main table if they differ from the example.

## Evaluation

Run the same test1000 metric scripts on these two directories:

```text
results/identity_source_ablation/stage1_source
results/identity_source_ablation/lq_source
```

Report PSNR, SSIM, LPIPS, ArcFace cosine, and AdaFace cosine. Preserve the
per-image CSV files so paired differences and confidence intervals can be
computed later.
