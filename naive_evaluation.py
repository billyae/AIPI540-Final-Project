import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# import your naive pipeline
from naive import learn_palette_cv2, apply_palette_and_edges

# ─────────────── PARAMETERS ───────────────
TRAIN_LABEL_DIR  = 'dataset/train/labels'
TEST_IMG_DIR     = 'dataset/test/images'
TEST_LABEL_DIR   = 'dataset/test/labels'
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
CLIP_BACKBONE    = 'openai/clip-vit-base-patch32'
# ──────────────────────────────────────────────

def stylize_image(path: str, palette: np.ndarray) -> Image.Image:
    """
    Read human image at `path`, apply palette+edges, convert to RGB PIL.
    Args:
        path (str): Path to the input image.
        palette (np.ndarray): BGR palette of shape (PALETTE_SIZE, 3).
    Returns:
        Image.Image: Stylized image in RGB format.
    """

    # load and convert to BGR
    human_bgr = cv2.imread(path)
    if human_bgr is None:
        raise RuntimeError(f"Failed to load {path}")
    
    # apply palette and edges
    out_bgr   = apply_palette_and_edges(human_bgr, palette)
    out_rgb   = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)

def clip_cosine(clip_model, clip_proc, img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute CLIP image‐to‐image cosine similarity.
    Args:
        clip_model (CLIPModel): Pretrained CLIP model.
        clip_proc (CLIPProcessor): Pretrained CLIP processor.
        img1 (Image.Image): First image.
        img2 (Image.Image): Second image.
    Returns:
        float: Cosine similarity between the two images.
    """

    batch = clip_proc(images=[img1, img2], return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        feats = clip_model.get_image_features(**batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return F.cosine_similarity(feats[0:1], feats[1:2]).item()

def main():
    """
    1) Learn a global palette from training labels.
    2) Load CLIP model.
    3) Iterate over test images, stylize them, and compute CLIP similarity with GT.
    4) Report average similarity.
    Args:
        None
    """
    # learn palette from training labels
    print("Learning palette from train labels…")
    palette = learn_palette_cv2(TRAIN_LABEL_DIR)

    # load CLIP
    print("Loading CLIP model…")
    clip_model     = CLIPModel.from_pretrained(CLIP_BACKBONE).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_BACKBONE)

    # iterate test images
    sims = []
    for img_path in tqdm(sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*'))), desc="Evaluating"):
        fn         = os.path.basename(img_path)
        label_path = os.path.join(TEST_LABEL_DIR, fn)
        if not os.path.exists(label_path):
            print(f"  → no ground truth for {fn}, skipping")
            continue

        # stylize and load GT
        stylized = stylize_image(img_path, palette)
        gt       = Image.open(label_path).convert('RGB')

        # compute CLIP similarity
        sim = clip_cosine(clip_model, clip_processor, stylized, gt)
        sims.append(sim)
        print(f"{fn}: {sim:.4f}")

    # report
    if sims:
        print(f"\nAverage CLIP similarity over {len(sims)} images: {np.mean(sims):.4f}")
    else:
        print("No images evaluated.")
    
if __name__ == '__main__':
    main()
    
