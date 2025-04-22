import os
import glob
import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# ─────────────── PARAMETERS ───────────────
MODEL_PATH       = 'rf_style_with_val_512.joblib'  # your trained model
TEST_IMAGE_DIR   = 'dataset/test/images'          # input (human) images
TEST_LABEL_DIR   = 'dataset/test/labels'          # ground‑truth (anime) images
PATCH_RADIUS     = 1                              # must match training
TARGET_SIZE      = (512, 512)                     # (width, height)
CLIP_MODEL_NAME  = 'openai/clip-vit-base-patch32' # CLIP backbone
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
# ──────────────────────────────────────────────

def extract_features(img: np.ndarray, patch_radius: int) -> np.ndarray:
    """
    Given H×W×3 image in [0,1], returns (H*W, (2r+1)^2*3) array of flattened RGB patches.
    """
    H, W, _ = img.shape
    r = patch_radius
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode='reflect')
    feats = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            patch = pad[r+dy : r+dy+H, r+dx : r+dx+W]  # H×W×3
            feats.append(patch.reshape(-1, 3))
    return np.concatenate(feats, axis=1)  # (H*W, (2r+1)^2*3)

def reconstruct_image(rf_model, img_path: str) -> Image.Image:
    """Load, resize, extract features, predict RGBs, and reassemble into a PIL image."""
    pil = Image.open(img_path).convert('RGB').resize(TARGET_SIZE, Image.BILINEAR)
    im = np.asarray(pil, dtype=np.float32) / 255.0
    X = extract_features(im, PATCH_RADIUS)
    pred = rf_model.predict(X)
    H, W = TARGET_SIZE[1], TARGET_SIZE[0]
    pred = pred.reshape(H, W, 3)
    pred = (pred * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pred)

def compute_clip_similarity(model, processor, img1: Image.Image, img2: Image.Image) -> float:
    """Returns cosine similarity between image embeddings from CLIP."""
    inputs = processor(
        images=[img1, img2],
        return_tensors='pt',
        padding=True
    ).to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    # normalize and compute cosine similarity
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return F.cosine_similarity(feats[0:1], feats[1:2]).item()

if __name__ == '__main__':
    # Load your RandomForest model
    rf = joblib.load(MODEL_PATH)

    # Load CLIP
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    # Gather test files
    test_imgs = sorted(glob.glob(os.path.join(TEST_IMAGE_DIR, '*')))
    sims = []

    for img_path in test_imgs:
        fn = os.path.basename(img_path)
        label_path = os.path.join(TEST_LABEL_DIR, fn)
        if not os.path.exists(label_path):
            print(f"Warning: no label for {fn}, skipping.")
            continue

        # Reconstruct and load ground truth
        pred_img = reconstruct_image(rf, img_path)
        gt_img   = Image.open(label_path).convert('RGB').resize(TARGET_SIZE, Image.BILINEAR)

        # Compute CLIP similarity
        sim = compute_clip_similarity(clip_model, clip_processor, pred_img, gt_img)
        sims.append(sim)
        print(f"{fn}: CLIP similarity = {sim:.4f}")

    if sims:
        print(f"\nAverage CLIP similarity over {len(sims)} images: {np.mean(sims):.4f}")
    else:
        print("No test images evaluated.")