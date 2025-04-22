import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor

# ─────────────── CONFIGURATION ───────────────
CHECKPOINT = "/content/drive/MyDrive/controlnet-finetuned/epoch-9"
TEST_DIR = "/content/drive/MyDrive/anime-style-transfer/test"
# Batch size for DataLoader
BATCH_SIZE = 4
# Resize height and width
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of diffusion steps
NUM_INFERENCE_STEPS = 20

class ImageLabelDataset(Dataset):
    """Dataset pairing each test image with its matching label (anime target)."""
    def __init__(self, images_dir: str, labels_dir: str, img_size: int):
        """
        Args:
            images_dir (str): Directory containing test images.
            labels_dir (str): Directory containing corresponding labels.
            img_size (int): Size to which images will be resized.
        """
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
        self.labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir)])
        assert len(self.images) == len(self.labels), (
            f"Number of images ({len(self.images)}) and labels ({len(self.labels)}) must match"
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        """

        # Load and transform image and label
        img = Image.open(self.images[idx]).convert("RGB")
        lbl = Image.open(self.labels[idx]).convert("RGB")
        return {
            "controlnet_cond": self.transform(img),
            "pixel_values": self.transform(lbl),
            "prompt": "a fantasy illustration"
        }


def validate(pipe, loader, clip_model, clip_processor, device):
    """
    Compute average cosine similarity between generated and ground-truth anime images using CLIP.
    Args:
        pipe (StableDiffusionControlNetPipeline): The pipeline to generate images.
        loader (DataLoader): DataLoader for the test dataset.
        clip_model (CLIPModel): CLIP model for feature extraction.
        clip_processor (CLIPProcessor): Processor for preparing inputs for CLIP.
        device (str): Device to run the evaluation on.
    Returns:
        float: Average cosine similarity score.
    """

    # Set models to evaluation mode
    pipe.unet.eval()
    clip_model.eval()
    sims = []

    # Iterate through the DataLoader
    to_pil = transforms.ToPILImage()
    for batch in loader:

        # Move batch to device
        prompts = batch["prompt"]
        cond_tensors = batch["controlnet_cond"].to(device)
        target_tensors = batch["pixel_values"].to(device)

        # Generate images
        cond_pils = [to_pil((t.cpu() * 0.5 + 0.5)) for t in cond_tensors]
        generated = pipe(
            prompt=prompts,
            image=cond_pils,
            num_inference_steps=NUM_INFERENCE_STEPS
        ).images

        # Prepare target PIL images
        target_pils = [to_pil((t.cpu() * 0.5 + 0.5)) for t in target_tensors]

        # CLIP inputs
        inputs = clip_processor(
            images=generated + target_pils,
            return_tensors="pt"
        ).to(device)

        # Feature extraction and cosine sim
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)

        # Normalize embeddings
        gen_emb, tgt_emb = emb.chunk(2, dim=0)
        batch_sims = F.cosine_similarity(gen_emb, tgt_emb, dim=-1)
        sims.append(batch_sims.cpu())

    # Concatenate all batch similarities and compute mean
    sims = torch.cat(sims)
    return sims.mean().item()


def main():
    """
    Main function to load the model, prepare the dataset, and run evaluation.
    Args:
        None
    """
    # Load fine-tuned SD+ControlNet pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        CHECKPOINT,
        safety_checker=None,
        torch_dtype=torch.float32
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load CLIP model & processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Build test DataLoader
    images_dir = os.path.join(TEST_DIR, "images")
    labels_dir = os.path.join(TEST_DIR, "labels")
    test_ds = ImageLabelDataset(images_dir, labels_dir, IMG_SIZE)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Run evaluation
    print("Running CLIP-based evaluation on test set…")
    test_score = validate(pipe, test_loader, clip_model, clip_processor, DEVICE)
    print(f"CLIP test similarity: {test_score:.4f}")


if __name__ == "__main__":
    main()
