import os
import requests
import numpy as np
import pandas as pd

from io import BytesIO
from PIL import Image
import mediapipe as mp
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def build_dataset(
    split: str = "train",
    num_images: int = 5000,
    shuffle_seed: int = 42,
    output_root: str = "dataset",
) -> None:
    """
    Fetches images from HF, computes MediaPipe pose overlays, and saves:

      DATA_ROOT/
      ├─ target/
      │   ├─ 000000.png
      │   ├─ 000001.png
      │   └─ …
      └─ pose/
          ├─ 000000.png
          ├─ 000001.png
          └─ …

    Args:
        split (str): HF dataset split to use.
        num_images (int): How many examples to download.
        shuffle_seed (int): Seed for shuffling.
        output_root (str): Directory under which `target/` and `pose/` will be created.
    """
    # 1) Load & filter
    ds = load_dataset("KBlueLeaf/danbooru2023-metadata-database", split=split)
    ds = ds.filter(lambda _, i: i != 34, with_indices=True)
    ds = ds.shuffle(seed=shuffle_seed).select(range(num_images))
    ds = ds.filter(
        lambda ex: ex["file_url"]
        and ex["file_url"].startswith("http")
        and ex["file_url"].lower().endswith((".png", ".jpg", ".jpeg"))
    )

    # 2) Prepare output folders
    target_dir = os.path.join(output_root, "target")
    pose_dir = os.path.join(output_root, "pose")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # 3) Helpers
    def fetch_and_resize(url: str) -> np.ndarray:
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = img.resize((512, 512), Image.LANCZOS)
        return np.array(img)

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    def compute_pose_map(img_np: np.ndarray) -> np.ndarray:
        results = pose.process(img_np)
        annotated = img_np.copy()
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
               annotated,
               results.pose_landmarks,
               mp_pose.POSE_CONNECTIONS,
               landmark_drawing_spec=mp_draw.DrawingSpec(thickness=2, circle_radius=2),
               connection_drawing_spec=mp_draw.DrawingSpec(thickness=2)
           )
        return annotated

    # 4) Download, annotate, and save
    for idx, example in enumerate(ds):
        filename = f"{idx:06d}.png"

        # target
        tgt = fetch_and_resize(example["file_url"])
        Image.fromarray(tgt).save(os.path.join(target_dir, filename))

        # pose_map
        pm = compute_pose_map(tgt)
        Image.fromarray(pm).save(os.path.join(pose_dir, filename))

        if idx and idx % 500 == 0:
            print(f"  → Saved {idx} images...")

    pose.close()
    print(f"Done! Saved {num_images} pairs under '{output_root}/'.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--split",       type=str,   default="train")
    p.add_argument("--num_images",  type=int,   default=5000)
    p.add_argument("--shuffle_seed",type=int,   default=42)
    p.add_argument("--output_root", type=str,   required=True,
                   help="Root dir to create `target/` and `pose/`")
    args = p.parse_args()

    build_dataset(
        split=args.split,
        num_images=args.num_images,
        shuffle_seed=args.shuffle_seed,
        output_root=args.output_root,
    )
