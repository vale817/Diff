import copy
import os
from argparse import ArgumentParser

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffbir.model import ControlLDM, Diffusion, SwinIR, build_identity_encoder
from diffbir.model.config import AttnMode, Config
from diffbir.utils.common import instantiate_from_config, to


def load_swinir(cfg, device) -> SwinIR:
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    state_dict = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    swinir.load_state_dict(state_dict, strict=True)
    swinir.eval().to(device)
    for parameter in swinir.parameters():
        parameter.requires_grad = False
    return swinir


def main(args) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    required_paths = [
        "sd_path",
        "controlnet_path",
        "swinir_path",
        "identity_encoder_path",
        "exp_dir",
    ]
    missing_paths = [name for name in required_paths if not cfg.train.get(name)]
    if not cfg.dataset.train.params.file_list:
        missing_paths.append("dataset.train.params.file_list")
    if missing_paths:
        raise ValueError(
            "missing required train config values: " + ", ".join(missing_paths)
        )

    exp_dir = cfg.train.exp_dir
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter(exp_dir)

    Config.attn_mode = AttnMode.SDP
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    cldm.load_pretrained_sd(sd)
    control = torch.load(cfg.train.controlnet_path, map_location="cpu")
    if "state_dict" in control:
        control = control["state_dict"]
    cldm.load_controlnet_from_ckpt(control)
    if cfg.train.identity_resume:
        identity = torch.load(cfg.train.identity_resume, map_location="cpu")
        if "state_dict" in identity:
            identity = identity["state_dict"]
        cldm.load_identity_state_dict(identity)
    cldm.set_identity_trainable()

    swinir = load_swinir(cfg, device)
    identity_encoder = build_identity_encoder(cfg.train.identity_encoder_path)
    identity_encoder.eval().to(device)
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion).to(device)

    trainable = list(cldm.identity_parameters())
    if accelerator.is_main_process:
        count = sum(parameter.numel() for parameter in trainable)
        print(f"Trainable identity parameters: {count:,}")
    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.learning_rate)
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    batch_transform = instantiate_from_config(cfg.batch_transform)

    cldm.eval().to(device)
    cldm.identity_projector.train()
    cldm, optimizer, loader = accelerator.prepare(cldm, optimizer, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)

    global_step = 0
    max_steps = cfg.train.train_steps
    progress = tqdm(
        total=max_steps,
        disable=not accelerator.is_main_process,
        desc="identity attention stage 1",
    )
    optimizer.zero_grad()
    while global_step < max_steps:
        for batch in loader:
            batch = to(batch, device)
            gt, lq, prompt = batch_transform(batch)
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            with torch.no_grad(), accelerator.autocast():
                z_0 = pure_cldm.vae_encode(gt)
                stage1 = swinir(lq)
                identity_embedding = identity_encoder(stage1.clamp(0, 1))
                cond = pure_cldm.prepare_condition(
                    stage1,
                    prompt,
                    identity_embedding=identity_embedding,
                    identity_scale=cfg.train.identity_scale,
                )
                cond_aug = copy.deepcopy(cond)
                if cfg.train.noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0,
                            cfg.train.noise_aug_timestep,
                            (z_0.size(0),),
                            device=device,
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )

            timestep = torch.randint(
                0, diffusion.num_timesteps, (z_0.size(0),), device=device
            )
            with accelerator.accumulate(cldm):
                loss = diffusion.p_losses(cldm, z_0, timestep, cond_aug)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.5f}")
                if accelerator.is_main_process:
                    writer.add_scalar("loss/diffusion", loss.item(), global_step)
                if global_step % cfg.train.ckpt_every == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        torch.save(
                            {
                                "state_dict": pure_cldm.identity_state_dict(),
                                "global_step": global_step,
                                "config": OmegaConf.to_container(cfg, resolve=True),
                            },
                            os.path.join(ckpt_dir, f"{global_step:07d}.pt"),
                        )
                if global_step >= max_steps:
                    break

    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(
            {
                "state_dict": pure_cldm.identity_state_dict(),
                "global_step": global_step,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            os.path.join(ckpt_dir, "last.pt"),
        )
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    main(parser.parse_args())
