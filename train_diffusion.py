import os
import random
from glob import glob
from dataclasses import dataclass
from pathlib import Path

from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from accelerate import Accelerator
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.amp import autocast, GradScaler

@dataclass
class TrainingConfig:
    seed: int = 42
    lr: float = 1e-4
    epochs: int = 10
    batch_size: int = 4
    img_size: int = 512
    scheduler_ckpt: str = "runwayml/stable-diffusion-v1-5"
    controlnet_ckpt: str = "lllyasviel/sd-controlnet-canny"
    output_dir: str = "./controlnet-finetuned"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr_warmup_steps: int = 100


def set_seed(seed: int):
    """Deterministically set all random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageLabelDataset(Dataset):
    """Pairs each image with its same-named label (control map)."""

    def __init__(self, images_dir: str, labels_dir: str, img_size: int):
        self.images = sorted(glob(f"{images_dir}/*.*"))
        self.labels = sorted(glob(f"{labels_dir}/*.*"))
        assert len(self.images) == len(self.labels), "Mismatch #images vs #labels"
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = Image.open(self.images[idx]).convert("RGB")
        lbl = Image.open(self.labels[idx]).convert("RGB")
        
        return {
            "pixel_values": self.transform(lbl),
            "controlnet_cond": self.transform(img),
            "prompt": "a fantasy illustration"  # replace with your own prompts
        }


def build_dataloader(config: TrainingConfig, images_dir: str, labels_dir: str):
    ds = ImageLabelDataset(images_dir, labels_dir, config.img_size)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


def load_controlnet_pipeline(config: TrainingConfig):
    """Load pretrained Stable Diffusion + ControlNet and swap in a DDIM scheduler."""
    controlnet = ControlNetModel.from_pretrained(
        config.controlnet_ckpt, torch_dtype=torch.float32
    ).to(config.device)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.scheduler_ckpt,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float32,
    ).to(config.device)

    # replace with a deterministic scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # freeze everything except UNet & ControlNet
    for p in pipe.text_encoder.parameters(): p.requires_grad = False
    for p in pipe.vae.parameters():        p.requires_grad = False
    for p in pipe.unet.parameters():       p.requires_grad = False

    return pipe, controlnet


def setup_optimizer_and_scheduler(pipe, config: TrainingConfig, total_steps: int):
    """Only train UNet + ControlNet parameters."""
    params = filter(lambda p: p.requires_grad, pipe.controlnet.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler


def train_one_epoch(pipe, loader, optimizer, scheduler, accelerator, config: TrainingConfig):
    """Run a single epoch of DDPM loss with ControlNet conditioning."""
    pipe.unet.train()
    for batch in loader:
        prompts    = batch["prompt"]

        pixel_vals = batch["pixel_values"].to(
            config.device, 
            dtype=pipe.vae.encoder.conv_in.weight.dtype  # or simply pipe.vae.dtype
        )

        cond_imgs = batch["controlnet_cond"].to(
            config.device, 
            dtype=pipe.controlnet.dtype
        )

        # 1) encode text
        tokenized = pipe.tokenizer(
            prompts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(config.device)
        encoder_states = pipe.text_encoder(**tokenized)[0]

        # 2) get latents
        latents = pipe.vae.encode(pixel_vals).latent_dist.sample() * 0.18215

        # 3) sample noise & timesteps
        timesteps = torch.randint(
            0,
            pipe.scheduler.config.num_train_timesteps,
            (pixel_vals.shape[0],),
            device=config.device,
            dtype=torch.long
        )
        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # 4) predict & backprop
        with accelerator.accumulate(pipe.unet):
            control_outputs = pipe.controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_states,
                controlnet_cond=cond_imgs,    # your edge maps / line art / whatever
                return_dict=False,            # gets back a tuple below
            )
            # unpack however your version returns them:
            down_block_residuals, mid_block_residual = control_outputs 

            # 3) feed those residuals into the UNet
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_states,
                down_block_additional_residuals=down_block_residuals,
                mid_block_additional_residual=mid_block_residual,
                return_dict=False,
            )[0]  # [0] to grab the tensor out of the returned tuple or dict

            # noise = noise.to(torch.float32)
            # noise_pred = noise_pred.to(torch.float16)
            # print(f"noise_pred dtype: {noise_pred.dtype}")
            # print(f"noise dtype: {noise.dtype}")
            # 4) compute your DDPM / MSE loss
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # scaler = GradScaler()

            # with autocast():
            #     noise_pred = pipe.unet(...)
            #     loss = F.mse_loss(noise_pred, noise)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

    return loss.item()


def save_checkpoint(pipe, epoch: int, output_dir: str, accelerator: Accelerator):
    """Save both UNet + ControlNet at each checkpoint."""
    if accelerator.is_main_process:
        ckpt_dir = Path(output_dir) / f"epoch-{epoch}"
        pipe.save_pretrained(ckpt_dir)


def main():
    cfg = TrainingConfig()
    set_seed(cfg.seed)

    # prepare data
    loader = build_dataloader(cfg, "dataset/train/images", "dataset/train/labels")
    total_steps = len(loader) * cfg.epochs

    # load models
    pipe, controlnet = load_controlnet_pipeline(cfg)

    
    optimizer, scheduler = setup_optimizer_and_scheduler(pipe, cfg, total_steps)

   
    # accelerator wraps models & optimizer & dataloader & scheduler
    accelerator = Accelerator(mixed_precision="fp16")
    pipe.unet, pipe.controlnet, optimizer, loader, scheduler = accelerator.prepare(
        pipe.unet, pipe.controlnet, optimizer, loader, scheduler
    )

    # training loop
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(pipe, loader, optimizer, scheduler, accelerator, cfg)
        print(f"[Epoch {epoch:02d}] loss: {loss:.4f}")
        save_checkpoint(pipe, epoch, cfg.output_dir, accelerator)


if __name__ == "__main__":
    main()
