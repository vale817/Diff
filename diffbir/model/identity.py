from typing import Any
import os

import torch
from torch import nn
from torch.nn import functional as F


class IdentityTokenProjector(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        context_dim: int = 1024,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_tokens * context_dim),
        )
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding = F.normalize(embedding, dim=-1)
        tokens = self.proj(embedding)
        tokens = tokens.view(-1, self.num_tokens, self.context_dim)
        return self.norm(tokens)


class TorchScriptIdentityEncoder(nn.Module):
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


class ONNXRuntimeIdentityEncoder(nn.Module):
    """Frozen ArcFace identity encoder backed by an ONNX model.

    This is intended for InsightFace buffalo_l/w600k_r50.onnx style models.
    The encoder is used under torch.no_grad(), so ONNXRuntime is acceptable for
    training even though it is not differentiable.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 112,
        input_mean: float = 0.5,
        input_std: float = 0.5,
    ) -> None:
        super().__init__()
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ONNX identity encoder requires onnxruntime or onnxruntime-gpu. "
                "Install it with: pip install onnxruntime-gpu"
            ) from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        if not providers:
            providers = available
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.input_mean = input_mean
        self.input_std = input_std

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device
        image = F.interpolate(
            image,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        image = (image - self.input_mean) / self.input_std
        image_np = image.detach().float().cpu().numpy()
        embedding = self.session.run(None, {self.input_name: image_np})[0]
        embedding = torch.from_numpy(embedding).to(device=device, dtype=torch.float32)
        if embedding.ndim > 2:
            embedding = embedding.flatten(1)
        return F.normalize(embedding, dim=-1)


def build_identity_encoder(model_path: str) -> nn.Module:
    """Build identity encoder from TorchScript (.pt/.pth) or ONNX (.onnx)."""
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".onnx":
        return ONNXRuntimeIdentityEncoder(model_path)
    return TorchScriptIdentityEncoder(model_path)
