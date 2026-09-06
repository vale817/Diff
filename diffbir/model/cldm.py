from typing import Tuple, Set, List, Dict

import torch
from torch import nn

from .controlnet import ControlledUnetModel, ControlNet
from .vae import AutoencoderKL
from .util import GroupNorm32
from .clip import FrozenOpenCLIPEmbedder
from .distributions import DiagonalGaussianDistribution
from .identity import IdentityTokenProjector
from .config import AttnMode, Config
from ..utils.tilevae import VAEHook


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):

    def __init__(
        self,
        unet_cfg,
        vae_cfg,
        clip_cfg,
        controlnet_cfg,
        latent_scale_factor,
        identity_projector_cfg=None,
    ):
        super().__init__()
        if identity_projector_cfg is not None and Config.attn_mode != AttnMode.SDP:
            raise RuntimeError(
                "stage-1 identity attention currently requires ATTN_MODE=sdp"
            )
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.identity_projector = (
            IdentityTokenProjector(**identity_projector_cfg)
            if identity_projector_cfg is not None
            else None
        )
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        missing = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch

    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            z = encoder(image).mode() * self.scale_factor
        return z

    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(z / self.scale_factor)

    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: List[str],
        tiled: bool = False,
        tile_size: int = -1,
        identity_embedding: torch.Tensor | None = None,
        identity_scale: float | torch.Tensor = 0.0,
    ) -> Dict[str, torch.Tensor]:
        condition = dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(
                cond_img * 2 - 1,
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
            ),
        )
        if identity_embedding is not None:
            if self.identity_projector is None:
                raise RuntimeError("identity projector is not configured")
            condition["c_id"] = identity_embedding
            if torch.is_tensor(identity_scale):
                scale = identity_scale.to(identity_embedding)
            else:
                scale = identity_embedding.new_full(
                    (identity_embedding.size(0),), identity_scale
                )
            condition["c_id_scale"] = scale
        return condition

    def forward(self, x_noisy, t, cond):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        control = self.controlnet(x=x_noisy, hint=c_img, timesteps=t, context=c_txt)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        unet_context = c_txt
        if "c_id" in cond:
            identity_scale = cond["c_id_scale"]
            if torch.any(identity_scale != 0):
                identity_tokens = self.identity_projector(cond["c_id"])
                unet_context = {
                    "text": c_txt,
                    "identity": identity_tokens,
                    "identity_scale": identity_scale,
                }
        eps = self.unet(
            x=x_noisy,
            timesteps=t,
            context=unet_context,
            control=control,
            only_mid_control=False,
        )
        return eps

    def set_identity_trainable(self) -> None:
        if self.identity_projector is None:
            raise RuntimeError("identity projector is not configured")
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.identity_projector.parameters():
            parameter.requires_grad = True
        found_attention = False
        for name, parameter in self.unet.named_parameters():
            if ".to_k_id." in name or ".to_v_id." in name:
                parameter.requires_grad = True
                found_attention = True
        if not found_attention:
            raise RuntimeError("UNet has no identity attention parameters")

    def identity_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def identity_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if name.startswith("identity_projector.")
            or ".to_k_id." in name
            or ".to_v_id." in name
        }

    def load_identity_state_dict(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = incompatible.unexpected_keys
        expected = set(self.identity_state_dict())
        missing = sorted(expected - set(state_dict))
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"invalid identity checkpoint, missing={missing}, "
                f"unexpected={unexpected}"
            )

    def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
        self.unet.dtype = dtype
        self.controlnet.dtype = dtype
        # convert unet blocks to dtype
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)
        # convert controlnet blocks and zero-convs to dtype
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.type(dtype)

        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)

        # GroupNorm32 only works with float32
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)
