import argparse
from pathlib import Path

import cv2
import lpips
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_lpips_tensor(img, device):
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img * 2.0 - 1.0
    return img.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    gt_files = sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    for gt_path in tqdm(gt_files):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            print(f"[skip] missing prediction: {pred_path}")
            continue

        gt = read_rgb(gt_path)
        pred = read_rgb(pred_path)

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
        ssim = structural_similarity(gt, pred, channel_axis=2, data_range=255)

        with torch.no_grad():
            lp = lpips_model(
                to_lpips_tensor(gt, device),
                to_lpips_tensor(pred, device)
            ).item()

        rows.append({
            "name": gt_path.name,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lp,
        })

    df = pd.DataFrame(rows)
    per_image_csv = out_dir / f"{args.method}_per_image.csv"
    df.to_csv(per_image_csv, index=False)

    summary = pd.DataFrame([{
        "method": args.method,
        "num": len(df),
        "psnr_mean": df["psnr"].mean(),
        "ssim_mean": df["ssim"].mean(),
        "lpips_mean": df["lpips"].mean(),
    }])

    summary_csv = out_dir / f"{args.method}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))
    print(f"\nSaved per-image metrics: {per_image_csv}")
    print(f"Saved summary metrics:   {summary_csv}")


if __name__ == "__main__":
    main()
