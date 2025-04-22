import os
import random
from glob import glob
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
)
from diffusers.optimization import get_cosine_schedule_with_warmup

# ** NEW IMPORTS for CLIP **
from transformers import CLIPProcessor, CLIPModel

@dataclass
class TrainingConfig:
    seed: int = 42
    lr: float = 1e-4
    epochs: int = 10
    batch_size: int = 4
    img_size: int = 512
    val_split: float = 0.1       # fraction to hold out for validation
    scheduler_ckpt: str = "runwayml/stable-diffusion-v1-5"
    controlnet_ckpt: str = "lllyasviel/sd-controlnet-canny"
    output_dir: str = "/content/drive/MyDrive/controlnet-finetuned"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr_warmup_steps: int = 100

def set_seed(seed: int):
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

    def __len__(self): return len(self.images)

    def __getitem__(self, idx: int):
        img = Image.open(self.images[idx]).convert("RGB")
        lbl = Image.open(self.labels[idx]).convert("RGB")
        return {
            "pixel_values": self.transform(lbl),
            "controlnet_cond": self.transform(img),
            "prompt": "a fantasy illustration"
        }

def build_dataloaders(cfg: TrainingConfig, images_dir: str, labels_dir: str):
    full_ds = ImageLabelDataset(images_dir, labels_dir, cfg.img_size)
    val_size = int(len(full_ds) * cfg.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader

def load_controlnet_pipeline(cfg: TrainingConfig):
    controlnet = ControlNetModel.from_pretrained(
        cfg.controlnet_ckpt, torch_dtype=torch.float32
    ).to(cfg.device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.scheduler_ckpt,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float32,
    ).to(cfg.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # freeze everything except ControlNet
    for p in pipe.text_encoder.parameters(): p.requires_grad = False
    for p in pipe.vae.parameters():        p.requires_grad = False
    for p in pipe.unet.parameters():       p.requires_grad = False
    return pipe, controlnet

def setup_optimizer_and_scheduler(pipe, cfg: TrainingConfig, total_steps: int):
    params = filter(lambda p: p.requires_grad, pipe.controlnet.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.lr_warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler

def train_one_epoch(pipe, loader, optimizer, scheduler, accelerator, cfg: TrainingConfig):
    pipe.unet.train()
    for batch in loader:
        prompts    = batch["prompt"]
        pixel_vals = batch["pixel_values"].to(cfg.device, dtype=pipe.vae.dtype)
        cond_imgs  = batch["controlnet_cond"].to(cfg.device, dtype=pipe.controlnet.dtype)

        tokenized = pipe.tokenizer(prompts, padding="max_length",
                                   truncation=True, return_tensors="pt").to(cfg.device)
        encoder_states = pipe.text_encoder(**tokenized)[0]
        latents = pipe.vae.encode(pixel_vals).latent_dist.sample() * 0.18215

        timesteps = torch.randint(
            0, pipe.scheduler.config.num_train_timesteps,
            (pixel_vals.shape[0],), device=cfg.device, dtype=torch.long
        )
        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        with accelerator.accumulate(pipe.unet):
            down, mid = pipe.controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_states,
                controlnet_cond=cond_imgs,
                return_dict=False,
            )
            noise_pred = pipe.unet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_states,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                return_dict=False,
            )[0]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return loss.item()

@torch.no_grad()
def validate(pipe, val_loader, clip_model, clip_processor, cfg: TrainingConfig):
    pipe.unet.eval()
    clip_model.eval()
    sims = []
    for batch in val_loader:
        prompts    = batch["prompt"]
        cond_imgs  = batch["controlnet_cond"].to(cfg.device)
        target_imgs_pil = [
            transforms.ToPILImage()(batch["pixel_values"][i].cpu() * 0.5 + 0.5)
            for i in range(len(batch["pixel_values"]))
        ]

        # 1) generate outputs
        generated = pipe(
            prompt=prompts,
            image=[transforms.ToPILImage()(img.cpu()) for img in cond_imgs],
            num_inference_steps=20
        ).images

        # 2) prepare for CLIP
        inputs = clip_processor(
            images=generated + target_imgs_pil,
            return_tensors="pt"
        ).to(cfg.device)

        # 3) get embeddings & compute cosine sims
        emb = clip_model.get_image_features(**inputs)
        # first half are generated, second half are targets
        gen_emb, tgt_emb = emb.chunk(2, dim=0)
        batch_sims = F.cosine_similarity(gen_emb, tgt_emb, dim=-1)
        sims.append(batch_sims.cpu())

    sims = torch.cat(sims)
    return sims.mean().item()

def save_checkpoint(pipe, epoch: int, output_dir: str, accelerator: Accelerator):
    if accelerator.is_main_process:
        ckpt_dir = Path(output_dir) / f"epoch-{epoch}"
        pipe.save_pretrained(ckpt_dir)

def main():
    cfg = TrainingConfig()
    set_seed(cfg.seed)

    # 1) Build train & validation loaders
    train_loader, val_loader = build_dataloaders(
        cfg, "/content/drive/MyDrive/anime-style-transfer/train/images", "/content/drive/MyDrive/anime-style-transfer/train/labels"
    )
    total_steps = len(train_loader) * cfg.epochs

    # 2) Load SD+ControlNet
    pipe, controlnet = load_controlnet_pipeline(cfg)

    # 3) Load CLIP once
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(cfg.device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 4) Optimizer, scheduler, accelerator
    optimizer, scheduler = setup_optimizer_and_scheduler(pipe, cfg, total_steps)
    accelerator = Accelerator(mixed_precision="fp16")
    pipe.unet, pipe.controlnet, optimizer, train_loader, scheduler = accelerator.prepare(
        pipe.unet, pipe.controlnet, optimizer, train_loader, scheduler
    )

    # 5) Training + validation loop
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(pipe, train_loader, optimizer, scheduler, accelerator, cfg)
        val_score = validate(pipe, val_loader, clip_model, clip_processor, cfg)
        print(f"[Epoch {epoch:02d}] train loss: {loss:.4f}  |  CLIP val sim: {val_score:.4f}")
        if epoch == 9:
            save_checkpoint(pipe, epoch, cfg.output_dir, accelerator)

if __name__ == "__main__":
    main()