import os
import streamlit as st
from PIL import Image

import gdown
import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler

# CONFIGURATION
FOLDER_URL = "https://drive.google.com/drive/folders/1-4nOQAnYfs60XVnwjQzaT3l37e7mjJJ6?usp=drive_link"
MODEL_DIR  = "epoch-9"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def download_model_folder():
    """
    Downloads the entire Google Drive folder into MODEL_DIR.
    """

    # Check if the folder already exists
    if not os.path.isdir(MODEL_DIR):
        gdown.download_folder(
            url=FOLDER_URL,
            output=MODEL_DIR,
            quiet=False,
            use_cookies=False
        )
    else:
        st.info("Model folder already exists.")


@st.cache_resource
def load_pipeline():
    """
    Loads the StableDiffusionControlNetPipeline from local MODEL_DIR
    and switches its scheduler to DDIM.
    """

    # Load the pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_DIR,
        safety_checker=None,
        torch_dtype=torch.float32,
    )

    # Switch to DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe.to(DEVICE)


def setup_page():
    """
    Sets up Streamlit page config and title/description.
    """

    # Set up Streamlit page
    st.set_page_config(page_title="Anime Style Transfer", layout="centered")
    st.title("Anime Style Transfer 🎨")
    st.markdown(
        """
        1. Click **Download Model Folder** to fetch the pretrained pipeline directory.  
        2. Upload a human photo.  
        3. Click **Generate** to see the anime‑style output.
        """
            )


def run_app():
    """
    Main app logic: download, load, upload, and inference.
    """

    # Check if the model folder is ready
    if "model_ready" not in st.session_state:
        st.session_state.model_ready = False

    # Download model folder if not already done
    if not st.session_state.model_ready:
        if st.button("📥 Download Model Folder"):
            with st.spinner("Downloading model folder..."):
                download_model_folder()
            st.session_state.model_ready = True
            st.success("✅ Model folder ready!")
            st.experimental_rerun()   # ← force a rerun so the rest of your app shows
        else:
            st.info("Please download the model folder first.")
        return  # wait for download

    st.success("✅ Model is ready to use!")
    pipe = load_pipeline()

    # File uploader
    uploaded_file = st.file_uploader("Upload a human image", type=["png","jpg","jpeg"])
    if not uploaded_file:
        return

    # Display the uploaded image
    input_img = Image.open(uploaded_file).convert("RGB")
    st.image(input_img, caption="Original Photo", use_column_width=True)

    # Generate button
    if st.button("✨ Generate Anime Style"):
        with st.spinner("Generating..."):
            result = pipe(
                prompt="a fantasy illustration",
                image=input_img,
                num_inference_steps=20
            ).images[0]

        # Display the generated image
        st.image(result, caption="Anime Style Result", use_column_width=True)


def main():
    setup_page()
    run_app()


if __name__ == "__main__":
    main()
