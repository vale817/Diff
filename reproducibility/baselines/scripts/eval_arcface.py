import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor


class TorchScriptIdentityEncoder(torch.nn.Module):
    """Frozen aligned-face identity encoder backed by a TorchScript model."""

    def __init__(
        self,
        model_path: str,
        input_size: int = 112,
        input_mean: float = 0.5,
        input_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = torch.jit.load(model_path, map_location="cpu").eval()
        self.input_size = input_size
        self.input_mean = input_mean
        self.input_std = input_std
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        image = (image - self.input_mean) / self.input_std
        embedding: Any = self.model(image)
        if isinstance(embedding, (tuple, list)):
            embedding = embedding[0]
        if isinstance(embedding, dict):
            embedding = next(iter(embedding.values()))
        if embedding.ndim > 2:
            embedding = embedding.flatten(1)
        return F.normalize(embedding.float(), dim=-1)


class OnnxIdentityEncoder:
    """ArcFace-compatible ONNX encoder, e.g. InsightFace buffalo_l/w600k_r50.onnx."""

    def __init__(self, model_path: str, device: torch.device) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for --arcface_onnx. "
                "Install onnxruntime-gpu on CUDA servers or onnxruntime on CPU."
            ) from exc

        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print("warning: CUDAExecutionProvider is not available; using CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image_np = image.detach().cpu().numpy()
        image_np = image_np.transpose(0, 2, 3, 1)
        image_np = image_np * 255.0
        image_np = (image_np - 127.5) / 127.5
        image_np = image_np.transpose(0, 3, 1, 2).astype(np.float32)
        embedding = self.session.run(None, {self.input_name: image_np})[0]
        embedding = torch.from_numpy(embedding).float()
        if embedding.ndim > 2:
            embedding = embedding.flatten(1)
        return F.normalize(embedding, dim=-1)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: Path, recursive: bool) -> dict[str, Path]:
    pattern = "**/*" if recursive else "*"
    images = {}
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.relative_to(root).with_suffix("").as_posix().lower()
        if key in images:
            raise ValueError(f"duplicate image key '{key}' under {root}")
        images[key] = path
    return images


def load_image(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    return to_tensor(image)


def load_batch(paths: list[Path], size: int, device: torch.device) -> torch.Tensor:
    images = [load_image(path, size) for path in paths]
    return torch.stack(images, dim=0).to(device)


@torch.no_grad()
def compute_cosines(
    encoder,
    restored_paths: list[Path],
    reference_paths: list[Path],
    size: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    scores = []
    for start in tqdm(range(0, len(restored_paths), batch_size), desc="ArcFace cosine"):
        end = start + batch_size
        restored = load_batch(restored_paths[start:end], size, device)
        reference = load_batch(reference_paths[start:end], size, device)
        restored_feature = encoder(restored.clamp(0, 1))
        reference_feature = encoder(reference.clamp(0, 1))
        cosine = F.cosine_similarity(restored_feature, reference_feature, dim=-1)
        scores.extend(cosine.cpu().tolist())
    return scores


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Evaluate identity preservation with ArcFace cosine similarity. "
            "Images are matched by the same relative file path without extension."
        )
    )
    encoder_group = parser.add_mutually_exclusive_group(required=True)
    encoder_group.add_argument(
        "--arcface_encoder",
        help="ArcFace TorchScript model path.",
    )
    encoder_group.add_argument(
        "--arcface_onnx",
        help="ArcFace ONNX model path, e.g. InsightFace buffalo_l/w600k_r50.onnx.",
    )
    parser.add_argument("--restored_dir", required=True, help="Directory of restored images.")
    parser.add_argument(
        "--reference_dir",
        required=True,
        help="Directory of GT/reference identity images.",
    )
    parser.add_argument(
        "--baseline_dir",
        default="",
        help="Optional original DiffBIR output directory for relative comparison.",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Optional path to save per-image ArcFace cosine results.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_size", type=int, default=112)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--improve_threshold",
        type=float,
        default=0.0,
        help="Delta threshold for counting improved samples.",
    )
    parser.add_argument(
        "--drop_threshold",
        type=float,
        default=0.03,
        help="A sample is counted as obviously degraded when delta <= -drop_threshold.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    restored_dir = Path(args.restored_dir)
    reference_dir = Path(args.reference_dir)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None

    restored_index = collect_images(restored_dir, args.recursive)
    reference_index = collect_images(reference_dir, args.recursive)
    baseline_index = collect_images(baseline_dir, args.recursive) if baseline_dir else None

    keys = sorted(key for key in restored_index if key in reference_index)
    if not keys:
        raise ValueError("no matched images found between restored_dir and reference_dir")
    missing_reference = sorted(set(restored_index) - set(reference_index))
    if missing_reference:
        print(f"warning: {len(missing_reference)} restored images have no reference match")

    restored_paths = [restored_index[key] for key in keys]
    reference_paths = [reference_index[key] for key in keys]

    if args.arcface_onnx:
        encoder = OnnxIdentityEncoder(args.arcface_onnx, device)
    else:
        encoder = TorchScriptIdentityEncoder(
            args.arcface_encoder,
            input_size=args.eval_size,
        ).eval().to(device)

    restored_scores = compute_cosines(
        encoder,
        restored_paths,
        reference_paths,
        args.eval_size,
        args.batch_size,
        device,
    )

    baseline_scores = None
    deltas = None
    if baseline_index is not None:
        restored_score_by_key = dict(zip(keys, restored_scores))
        baseline_keys = [key for key in keys if key in baseline_index]
        if len(baseline_keys) != len(keys):
            print(f"warning: {len(keys) - len(baseline_keys)} samples have no baseline match")
        keys = baseline_keys
        restored_scores = [restored_score_by_key[key] for key in keys]
        baseline_paths = [baseline_index[key] for key in keys]
        reference_paths = [reference_index[key] for key in keys]
        baseline_scores = compute_cosines(
            encoder,
            baseline_paths,
            reference_paths,
            args.eval_size,
            args.batch_size,
            device,
        )
        deltas = [new - base for new, base in zip(restored_scores, baseline_scores)]

    print(f"samples: {len(keys)}")
    print(f"mean ArcFace cosine: {mean(restored_scores):.6f}")

    if baseline_scores is not None and deltas is not None:
        improved = sum(delta > args.improve_threshold for delta in deltas) / len(deltas)
        obvious_drop = sum(delta <= -args.drop_threshold for delta in deltas) / len(deltas)
        print(f"baseline mean ArcFace cosine: {mean(baseline_scores):.6f}")
        print(f"mean delta ArcFace: {mean(deltas):+.6f}")
        print(f"identity improved ratio: {improved:.2%}")
        print(f"obvious identity drop ratio: {obvious_drop:.2%}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as csv_file:
            fieldnames = ["key", "arcface_cosine"]
            if baseline_scores is not None:
                fieldnames += ["baseline_arcface_cosine", "delta_arcface"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, key in enumerate(keys):
                row = {
                    "key": key,
                    "arcface_cosine": restored_scores[index],
                }
                if baseline_scores is not None and deltas is not None:
                    row["baseline_arcface_cosine"] = baseline_scores[index]
                    row["delta_arcface"] = deltas[index]
                writer.writerow(row)
        print(f"saved per-image results to {output_path}")


if __name__ == "__main__":
    main()
