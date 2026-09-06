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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POSITIVE_LABELS = {"1", "true", "same", "positive", "pos", "genuine"}
NEGATIVE_LABELS = {"0", "false", "different", "diff", "negative", "neg", "imposter"}


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


def load_image(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    return to_tensor(image)


def label_from_token(token: str) -> int | None:
    label = token.strip().lower()
    if label in POSITIVE_LABELS:
        return 1
    if label in NEGATIVE_LABELS:
        return 0
    return None


def resolve_image(path_or_stem: str, image_root: Path) -> Path:
    raw_path = Path(path_or_stem)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(image_root / raw_path)

    if raw_path.suffix:
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    else:
        expanded = []
        for candidate in candidates:
            expanded.append(candidate)
            for extension in IMAGE_EXTENSIONS:
                expanded.append(candidate.with_suffix(extension))
        for candidate in expanded:
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(f"could not resolve image '{path_or_stem}' under {image_root}")


def resolve_lfw_image(identity: str, index: str, image_root: Path) -> Path:
    number = int(index)
    stem = f"{identity}_{number:04d}"
    candidates = []
    for extension in IMAGE_EXTENSIONS:
        candidates.append(image_root / identity / f"{stem}{extension}")
        candidates.append(image_root / f"{stem}{extension}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not resolve LFW-style image '{identity} {index}' under {image_root}"
    )


def split_pair_line(line: str) -> list[str]:
    if "," in line:
        return [token.strip() for token in next(csv.reader([line])) if token.strip()]
    return line.split()


def load_pairs(pair_file: Path, image_root: Path) -> list[tuple[Path, Path, int]]:
    pairs = []
    with pair_file.open() as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = split_pair_line(line)
            if line_number == 1 and len(tokens) in {1, 2} and all(token.isdigit() for token in tokens):
                continue

            first_label = label_from_token(tokens[0]) if tokens else None
            last_label = label_from_token(tokens[-1]) if tokens else None
            if first_label is not None and len(tokens) >= 3:
                image1 = resolve_image(tokens[1], image_root)
                image2 = resolve_image(tokens[2], image_root)
                pairs.append((image1, image2, first_label))
                continue
            if last_label is not None and len(tokens) >= 3:
                image1 = resolve_image(tokens[0], image_root)
                image2 = resolve_image(tokens[1], image_root)
                pairs.append((image1, image2, last_label))
                continue

            if len(tokens) == 3 and tokens[1].isdigit() and tokens[2].isdigit():
                image1 = resolve_lfw_image(tokens[0], tokens[1], image_root)
                image2 = resolve_lfw_image(tokens[0], tokens[2], image_root)
                pairs.append((image1, image2, 1))
                continue
            if (
                len(tokens) == 4
                and tokens[1].isdigit()
                and tokens[3].isdigit()
            ):
                image1 = resolve_lfw_image(tokens[0], tokens[1], image_root)
                image2 = resolve_lfw_image(tokens[2], tokens[3], image_root)
                pairs.append((image1, image2, 0))
                continue

            raise ValueError(
                f"unsupported pair format at {pair_file}:{line_number}: {raw_line.rstrip()}"
            )

    if not pairs:
        raise ValueError(f"no pairs were loaded from {pair_file}")
    return pairs


@torch.no_grad()
def compute_embeddings(
    encoder,
    paths: list[Path],
    size: int,
    batch_size: int,
    device: torch.device,
) -> dict[Path, torch.Tensor]:
    embeddings = {}
    unique_paths = sorted(set(paths))
    for start in tqdm(range(0, len(unique_paths), batch_size), desc="Identity embeddings"):
        batch_paths = unique_paths[start:start + batch_size]
        images = torch.stack([load_image(path, size) for path in batch_paths], dim=0).to(device)
        features = encoder(images.clamp(0, 1))
        for path, feature in zip(batch_paths, features.cpu()):
            embeddings[path] = feature
    return embeddings


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative pairs")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end

    positive_rank_sum = ranks[positives].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def best_verification_accuracy(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())

    tp = 0
    fp = 0
    best_accuracy = negative_count / len(labels)
    best_threshold = float(np.nextafter(sorted_scores[0], np.inf))
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        tp += int((group == 1).sum())
        fp += int((group == 0).sum())
        tn = negative_count - fp
        fn = positive_count - tp
        accuracy = (tp + tn) / len(labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(sorted_scores[start])
        start = end
    return float(best_accuracy), best_threshold


def tar_at_far(labels: np.ndarray, scores: np.ndarray, target_far: float) -> tuple[float, float, float]:
    thresholds = np.unique(scores)
    best_tar = 0.0
    best_far = 0.0
    best_threshold = float(np.nextafter(scores.max(), np.inf))
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())

    for threshold in thresholds:
        predicted_positive = scores >= threshold
        false_accepts = int(((labels == 0) & predicted_positive).sum())
        true_accepts = int(((labels == 1) & predicted_positive).sum())
        far = false_accepts / max(negative_count, 1)
        tar = true_accepts / max(positive_count, 1)
        if far <= target_far and tar >= best_tar:
            best_tar = tar
            best_far = far
            best_threshold = float(threshold)

    return float(best_tar), float(best_far), best_threshold


def evaluate_xqlfw(
    encoder,
    image_root: Path,
    pair_file: Path,
    eval_size: int,
    batch_size: int,
    device: torch.device,
    far_targets: list[float],
) -> tuple[dict[str, float], list[tuple[Path, Path, int, float]]]:
    pairs = load_pairs(pair_file, image_root)
    all_paths = [path for pair in pairs for path in pair[:2]]
    embeddings = compute_embeddings(encoder, all_paths, eval_size, batch_size, device)

    labels = []
    scores = []
    scored_pairs = []
    for image1, image2, label in pairs:
        score = F.cosine_similarity(embeddings[image1], embeddings[image2], dim=0).item()
        labels.append(label)
        scores.append(score)
        scored_pairs.append((image1, image2, label, score))

    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)
    accuracy, threshold = best_verification_accuracy(labels_np, scores_np)

    metrics = {
        "pairs": float(len(pairs)),
        "positive_pairs": float((labels_np == 1).sum()),
        "negative_pairs": float((labels_np == 0).sum()),
        "accuracy": accuracy,
        "best_threshold": threshold,
        "roc_auc": roc_auc_score(labels_np, scores_np),
    }
    for far in far_targets:
        tar, actual_far, tar_threshold = tar_at_far(labels_np, scores_np, far)
        metrics[f"tar@far={far:g}"] = tar
        metrics[f"actual_far@far={far:g}"] = actual_far
        metrics[f"threshold@far={far:g}"] = tar_threshold
    return metrics, scored_pairs


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Evaluate XQLFW/LFW-style face verification with cosine scores, "
            "reporting verification accuracy, ROC-AUC, and TAR@FAR."
        )
    )
    encoder_group = parser.add_mutually_exclusive_group(required=True)
    encoder_group.add_argument("--arcface_encoder", help="ArcFace TorchScript model path.")
    encoder_group.add_argument(
        "--arcface_onnx",
        help="ArcFace ONNX model path, e.g. InsightFace buffalo_l/w600k_r50.onnx.",
    )
    parser.add_argument("--image_root", required=True, help="Directory containing XQLFW images.")
    parser.add_argument("--pairs", required=True, help="XQLFW/LFW pair protocol file.")
    parser.add_argument("--output_csv", default="", help="Optional path for per-pair scores.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_size", type=int, default=112)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--far",
        type=float,
        nargs="+",
        default=[1e-2, 1e-3],
        help="FAR operating points for TAR reporting.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.arcface_onnx:
        encoder = OnnxIdentityEncoder(args.arcface_onnx, device)
    else:
        encoder = TorchScriptIdentityEncoder(
            args.arcface_encoder,
            input_size=args.eval_size,
        ).eval().to(device)

    metrics, scored_pairs = evaluate_xqlfw(
        encoder=encoder,
        image_root=Path(args.image_root),
        pair_file=Path(args.pairs),
        eval_size=args.eval_size,
        batch_size=args.batch_size,
        device=device,
        far_targets=args.far,
    )

    print(f"pairs: {int(metrics['pairs'])}")
    print(f"positive pairs: {int(metrics['positive_pairs'])}")
    print(f"negative pairs: {int(metrics['negative_pairs'])}")
    print(f"verification accuracy: {metrics['accuracy']:.6f}")
    print(f"best threshold: {metrics['best_threshold']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    for far in args.far:
        print(
            f"TAR@FAR={far:g}: {metrics[f'tar@far={far:g}']:.6f} "
            f"(actual FAR={metrics[f'actual_far@far={far:g}']:.6f}, "
            f"threshold={metrics[f'threshold@far={far:g}']:.6f})"
        )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["image1", "image2", "label", "cosine"],
            )
            writer.writeheader()
            for image1, image2, label, score in scored_pairs:
                writer.writerow(
                    {
                        "image1": image1.as_posix(),
                        "image2": image2.as_posix(),
                        "label": label,
                        "cosine": score,
                    }
                )
        print(f"saved per-pair scores to {output_path}")


if __name__ == "__main__":
    main()
