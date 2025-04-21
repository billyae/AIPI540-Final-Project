import os
import glob
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

def extract_features(img, patch_radius=1):
    """
    Given a H×W×3 image (as a numpy array in [0,1]),
    returns an array of shape (H*W, (2r+1)^2*3) with
    flattened RGB neighbourhoods around each pixel.
    """
    H, W, _ = img.shape
    r = patch_radius
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode='reflect')
    feats = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            patch = pad[r+dy : r+dy+H, r+dx : r+dx+W]  # H×W×3
            feats.append(patch.reshape(-1, 3))
    X = np.concatenate(feats, axis=1)  # (H*W, (2r+1)^2*3)
    return X


def build_dataset(
    image_dir,
    label_dir,
    patch_radius=1,
    sample_fraction=1.0,
    target_size=(512, 512)
):
    """
    Walk images/labels, extract features and targets.
    Optionally resizes images and labels to `target_size`.
    To keep memory down, we randomly sample a fraction of pixels.
    """
    all_X, all_y = [], []
    img_paths = sorted(glob.glob(os.path.join(image_dir, '*')))
    lbl_paths = sorted(glob.glob(os.path.join(label_dir, '*')))
    assert len(img_paths) == len(lbl_paths), "Number of images and labels must match"

    for imp, lp in zip(img_paths, lbl_paths):
        # Load with PIL and resize to target_size
        pil_im = Image.open(imp).convert('RGB')
        pil_lb = Image.open(lp).convert('RGB')
        pil_im = pil_im.resize(target_size, Image.BILINEAR)
        pil_lb = pil_lb.resize(target_size, Image.BILINEAR)

        # Convert to numpy arrays
        im = np.asarray(pil_im, dtype=np.float32) / 255.0
        lb = np.asarray(pil_lb, dtype=np.float32) / 255.0

        # Extract features and reshape labels
        X = extract_features(im, patch_radius)
        y = lb.reshape(-1, 3)

        # Randomly sample a subset of pixels
        idx = np.random.choice(
            X.shape[0],
            size=int(X.shape[0] * sample_fraction),
            replace=False
        )
        all_X.append(X[idx])
        all_y.append(y[idx])

    X_all = np.vstack(all_X)
    y_all = np.vstack(all_y)
    return X_all, y_all


def train(
    data_root: str = 'dataset/train',
    patch_radius: int = 1,
    sample_frac: float = 0.05,
    validation_frac: float = 0.2,
    n_estimators: int = 50,
    max_depth: int = 20,
    random_seed: int = 42,
    model_out: str = 'rf_style_with_val.joblib',
    target_size: tuple = (512, 512)
):
    """
    Train a RandomForestRegressor to map human-image patches → anime-image RGBs,
    and evaluate on a held-out validation set. Resizes data to `target_size`.

    Args:
        data_root:       path containing 'images/' and 'labels/' subfolders
        patch_radius:    neighborhood radius for feature extraction
        sample_frac:     fraction of pixels to sample per image
        validation_frac: fraction of data to reserve for validation
        n_estimators:    number of trees in the forest
        max_depth:       maximum tree depth
        random_seed:     random seed for reproducibility
        model_out:       filepath to save the trained model
        target_size:     (width, height) to resize images and labels
    """
    img_dir = os.path.join(data_root, 'images')
    lbl_dir = os.path.join(data_root, 'labels')

    print(f"[1/4] Building dataset from {img_dir} & {lbl_dir} at {target_size}…")
    X, y = build_dataset(
        image_dir=img_dir,
        label_dir=lbl_dir,
        patch_radius=patch_radius,
        sample_fraction=sample_frac,
        target_size=target_size
    )
    print(f"Total samples: {X.shape[0]:,}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=validation_frac,
        random_state=random_seed
    )
    print(f"[2/4] Training samples: {X_train.shape[0]:,}, Validation samples: {X_val.shape[0]:,}")

    print(f"[3/4] Training RandomForestRegressor ({n_estimators} trees, max_depth={max_depth})…")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        n_jobs=-1,
        verbose=1
    )
    rf.fit(X_train, y_train)

    print("[4/4] Evaluating on validation set…")
    y_pred = rf.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"Validation MSE: {mse:.6f}")
    print(f"Validation R^2: {r2:.4f}")

    dump(rf, model_out)
    print(f"Model saved to '{model_out}'. Training complete.")


if __name__ == '__main__':
    train(
        data_root='dataset/train',
        patch_radius=1,
        sample_frac=0.05,
        validation_frac=0.2,
        n_estimators=50,
        max_depth=20,
        random_seed=42,
        model_out='rf_style_with_val_512.joblib',
        target_size=(512, 512)
    )
