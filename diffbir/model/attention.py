from packaging import version
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from .util import checkpoint, zero_module, exists, default
from .config import Config, AttnMode


# CrossAttn precision handling
import os

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (vanilla). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            # with torch.autocast(enabled=False, device_type = 'cuda'):
            with torch.autocast(
                enabled=False,
                device_type="cuda" if str(x.device).startswith("cuda") else "cpu",
            ):
                q, k = q.float(), k.float()
                sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (xformers). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = Config.xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class SDPCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        identity_context_dim=None,
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (sdp). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_id = None
        self.to_v_id = None
        if identity_context_dim is not None:
            self.to_k_id = nn.Linear(identity_context_dim, inner_dim, bias=False)
            self.to_v_id = nn.Linear(identity_context_dim, inner_dim, bias=False)
            nn.init.normal_(self.to_k_id.weight, std=0.01)
            nn.init.zeros_(self.to_v_id.weight)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def _attention(self, q, context, to_k, to_v, batch_size):
        k = to_k(context)
        v = to_v(context)
        k, v = map(
            lambda tensor: tensor.unsqueeze(3)
            .reshape(batch_size, tensor.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.heads, tensor.shape[1], self.dim_head)
            .contiguous(),
            (k, v),
        )
        return F.scaled_dot_product_attention(q, k, v)

    def forward(self, x, context=None, mask=None):
        if exists(mask):
            raise NotImplementedError
        q = self.to_q(x)
        identity_context = None
        identity_scale = None
        if isinstance(context, dict):
            identity_context = context.get("identity")
            identity_scale = context.get("identity_scale")
            context = context["text"]
        context = default(context, x)

        b, _, _ = q.shape
        q = (
            q.unsqueeze(3)
            .reshape(b, q.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, q.shape[1], self.dim_head)
            .contiguous()
        )
        out = self._attention(q, context, self.to_k, self.to_v, b)

        if identity_context is not None:
            if self.to_k_id is None or self.to_v_id is None:
                raise RuntimeError("identity attention is not enabled for this block")
            out_id = self._attention(
                q, identity_context, self.to_k_id, self.to_v_id, b
            )
            out_id = (
                out_id.unsqueeze(0)
                .reshape(b, self.heads, out_id.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out_id.shape[1], self.heads * self.dim_head)
            )
            if identity_scale is not None:
                out_id = out_id * identity_scale.to(out_id).view(b, 1, 1)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if identity_context is not None:
            out = out + out_id
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        AttnMode.VANILLA: CrossAttention,  # vanilla attention
        AttnMode.XFORMERS: MemoryEfficientCrossAttention,
        AttnMode.SDP: SDPCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        identity_context_dim=None,
    ):
        super().__init__()
        attn_cls = self.ATTENTION_MODES[Config.attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            **(
                {"identity_context_dim": identity_context_dim}
                if attn_cls is SDPCrossAttention
                else {}
            ),
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        if isinstance(context, dict):
            return checkpoint(
                self._forward_identity,
                (
                    x,
                    context["text"],
                    context["identity"],
                    context["identity_scale"],
                ),
                self.parameters(),
                self.checkpoint,
            )
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward_identity(self, x, text, identity, identity_scale):
        return self._forward(
            x,
            {
                "text": text,
                "identity": identity,
                "identity_scale": identity_scale,
            },
        )

    def _forward(self, x, context=None):
        x = (
            self.attn1(
                self.norm1(x), context=context if self.disable_self_attn else None
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        identity_context_dim=None,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    identity_context_dim=identity_context_dim,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
