# Anime Character Generation

## Problem Statement

With the growing popularity of anime as a beloved art form and visual language, there’s a rising demand for tools that let anyone—from casual hobbyists to professional artists—easily transform real‑world photographs into vivid, stylized anime imagery. This project aims to bridge the gap between ordinary portrait photos and expressive anime art by developing and comparing three approaches: a traditional machine‑learning pipeline, a lightweight color‑quantization method, and a fine‑tuned diffusion model with ControlNet. By automating the transfer process, we hope to empower users to create unique anime‑style avatars and illustrations without requiring advanced artistic skills or manual editing.

## Link to the Demo

Generete Anime Character with Naive Approach: 
https://huggingface.co/spaces/billyae/anime-style-transfer-naive

Generate Anime Character with Deep Learning Approach:
https://huggingface.co/spaces/billyae/anime-style-transfer

## Link to the Presenation Video

https://www.youtube.com/watch?v=nMQfaxAPlp4&feature=youtu.be

## Data Source

All images come from the “Anime Style Transfer” dataset by Vincent V on Kaggle (https://www.kaggle.com/datasets/vincentv/anime-style-transfer). It provides paired real‑world human photos and their corresponding anime‑style renderings.

## Models Introduction

### Naive Color Quantization + Edges

- Learns a global palette via K‑means clustering on anime labels.
- Quantizes each human image pixel to nearest palette color and overlays Canny edges ​

### Traditional Machine Learning

- Extracts flattened RGB patches around each pixel.
- Trains a RandomForestRegressor to predict anime‑style RGB values per pixel ​

### Diffusion with ControlNet

- Fine‑tunes a Stable Diffusion pipeline augmented with ControlNet on human–anime pairs.

- Freezes text encoder, VAE, and U‑Net except ControlNet for efficient training.

### Model Links

- Tradtional_ml: https://drive.google.com/file/d/1Flikn3Gpnz8wMkX19hUQ4Npi1sJED0la/view?usp=sharing
- Deep Learning: https://drive.google.com/drive/folders/1-4nOQAnYfs60XVnwjQzaT3l37e7mjJJ6?usp=drive_link

## Evaluation Strategy

To quantitatively assess each model’s style‑transfer performance, we compute the CLIP‑based cosine similarity between generated images and their ground‑truth anime counterparts. CLIP (Contrastive Language–Image Pretraining) embeds images into a shared semantic space—images with similar visual content and style yield embeddings with high cosine similarity. By averaging this score over the test set, we obtain a robust, perception‑driven metric of how well each approach captures the target anime style.

Evaluation Results:  
- Naive Approach: 0.7241
- Traditional Approach: 0.7030
- Deep Learning Approach: 0.7827

## How to Run

### Set up Environments

`pip install requirements.txt`
### Training

- Naive Approach:  
`python naive_train.py`
- Traditional ML:  
`python traditional_ml_train.py`
- Diffusion:  
`python diffusion_train.py`

### Evaluation

- Naive Approach:  
`python naive_evaluation.py`
- Traditional ML:  
`python traditional_ml_evaluation.py`
- Diffusion:  
`python diffusion_evaluation.py`

### Demo Applications

- Naive Streamlit App: `streamlit run app_naive.py`
- Diffusion Streamlit App: `streamlit run app_dl.py`




