import os
import cv2
import numpy as np
from tqdm import tqdm

DATA_ROOT       = "./dataset/train"   # path to train/{images,labels}
OUTPUT_DIR      = "./outputs"         # where to save stylized outputs
PALETTE_SIZE    = 32                  # number of colours in the learned palette
SAMPLES_PER_IMG = 2000                # how many pixels to sample per anime image
EDGE_THRESH1    = 100                 # Canny lower threshold
EDGE_THRESH2    = 200                 # Canny upper threshold

def learn_palette_cv2(label_dir):
    """
    Build a global colour palette by sampling pixels from all anime labels
    and clustering with OpenCV's kmeans.
    Returns an array of shape (PALETTE_SIZE, 3) in BGR order (uint8).
    Args:
        label_dir (str): Directory containing anime labels.
    Returns:
        np.ndarray: Palette of shape (PALETTE_SIZE, 3).
    """

    # Sample pixels from all images
    samples = []
    for fn in tqdm(sorted(os.listdir(label_dir)), desc="Sampling anime pixels"):
        path = os.path.join(label_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        pixels = img.reshape(-1, 3)
        if len(pixels) > SAMPLES_PER_IMG:
            idx = np.random.choice(len(pixels), SAMPLES_PER_IMG, replace=False)
            pixels = pixels[idx]
        samples.append(pixels)
    if not samples:
        raise RuntimeError(f"No valid images found in {label_dir}")
    data = np.vstack(samples).astype(np.float32)

    # criteria: max 20 iters or epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    _, _, centers = cv2.kmeans(
        data, PALETTE_SIZE, None,
        criteria, 10, flags
    )
    palette = centers.astype(np.uint8)  # BGR centres
    return palette

def apply_palette_and_edges(human_bgr, palette):
    """
    1) Quantize each pixel in human_bgr to its nearest palette colour.
    2) Overlay Canny edges (drawn in black).
    Args:
        human_bgr (np.ndarray): Input image in BGR format.
        palette (np.ndarray): Colour palette of shape (PALETTE_SIZE, 3).
    Returns:
        np.ndarray: Stylized image with quantized colours and edges.
    """
    h, w = human_bgr.shape[:2]
    flat = human_bgr.reshape(-1, 3).astype(np.int32)  # N×3

    # Compute squared distances: (a−b)^2 = a^2 + b^2 − 2ab
    a2 = np.sum(flat * flat, axis=1, keepdims=True)       # N×1
    b2 = np.sum(palette * palette, axis=1)                # K
    ab = flat.dot(palette.T)                              # N×K
    dists = a2 + b2 - 2 * ab                              # N×K

    nearest = np.argmin(dists, axis=1)                    # N
    quant = palette[nearest].reshape(h, w, 3)

    # Overlay edges
    gray  = cv2.cvtColor(human_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=EDGE_THRESH1, threshold2=EDGE_THRESH2)

    # Make edges black
    stylized = quant.copy()
    stylized[edges > 0] = (0, 0, 0)
    return stylized

def main():
    """
    1) Learn a global palette from training labels.
    2) Iterate over human images, stylize them, and save the outputs.
    3) Report completion.
    Args:
        None
    """
    # Check directories
    img_dir   = os.path.join(DATA_ROOT, "images")
    label_dir = os.path.join(DATA_ROOT, "labels")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Learn the palette
    print(">> Learning colour palette from anime labels …")
    palette = learn_palette_cv2(label_dir)

    # Stylize each human image
    print(">> Stylizing human images …")
    for fn in tqdm(sorted(os.listdir(img_dir)), desc="Stylizing"):
        human_path = os.path.join(img_dir, fn)
        if not os.path.isfile(human_path):
            continue
        human = cv2.imread(human_path)
        if human is None:
            continue

        out = apply_palette_and_edges(human, palette)
        cv2.imwrite(os.path.join(OUTPUT_DIR, fn), out)

    print(f"\n✅ Done! Stylized images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
