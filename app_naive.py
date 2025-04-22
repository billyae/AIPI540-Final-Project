import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# import your functions from naive.py
from naive import learn_palette_cv2, apply_palette_and_edges

# ──────────────────────────────────────────────────────────────────────────────
LABEL_DIR = "./labels"
PALETTE_SIZE = None  # if you want to override inside naive.learn_palette_cv2

def load_palette(label_dir: str) -> np.ndarray:
    """
    Learn & return a (PALETTE_SIZE×3) BGR palette from label_dir.
    Each label is a 3-channel image with RGB values.
    The palette is learned by clustering the RGB values of all labels.
    Args:
        label_dir (str): Directory containing anime labels.
    Returns:    
        np.ndarray: Palette of shape (PALETTE_SIZE, 3).
    """
    return learn_palette_cv2(label_dir)


def stylize_image(img_pil: Image.Image, palette: np.ndarray) -> Image.Image:
    """
    Given a PIL image and a BGR palette, return a new PIL image
    with quantized colors + edge overlay.
    Args:
        img_pil (Image.Image): Input image as a PIL Image.
        palette (np.ndarray): BGR palette of shape (PALETTE_SIZE, 3).
    Returns:
        Image.Image: Stylized image as a PIL Image.
    """
    # convert to BGR for OpenCV
    human_rgb = np.array(img_pil)
    human_bgr = cv2.cvtColor(human_rgb, cv2.COLOR_RGB2BGR)

    # apply style
    stylized_bgr = apply_palette_and_edges(human_bgr, palette)

    # back to RGB for display
    stylized_rgb = cv2.cvtColor(stylized_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(stylized_rgb)


def main():
    """
    Streamlit app for naive anime style transfer.
    This app allows users to upload a human photo and apply
    a learned anime style transfer using a global color palette.
    The palette is learned from a folder of anime labels.
    The app uses OpenCV for image processing and Streamlit for the UI.
    """

    # Set up Streamlit app
    st.set_page_config(page_title="Naive Anime Style Transfer", layout="centered")
    st.title("Naive Anime Style Transfer 🎨")
    st.markdown(
        """
        This demo learns a global color palette from a folder of anime labels  
        and then quantizes + edge‑overlays your uploaded photo into an anime look.
        """
    )

    # Check if the label directory exists
    if not os.path.isdir(LABEL_DIR):
        st.error(f"Could not find your anime labels folder:\n**{LABEL_DIR}**")
        return

    # Compute palette once and cache
    @st.cache_data(show_spinner=False)
    def _get_palette():
        return load_palette(LABEL_DIR)

    with st.spinner("Learning colour palette from anime labels…"):
        palette = _get_palette()
    st.success("Palette ready!")

    # File uploader
    uploaded = st.file_uploader("Upload a human photo", type=["png", "jpg", "jpeg"])
    if not uploaded:
        return

    # Display & stylize
    img_pil = Image.open(uploaded).convert("RGB")
    st.image(img_pil, caption="Original Photo", use_container_width=True)

    with st.spinner("Applying palette & edges…"):
        result_pil = stylize_image(img_pil, palette)
    
    st.image(result_pil, caption="Anime‑Style Result", use_container_width=True)


if __name__ == "__main__":
    main()
