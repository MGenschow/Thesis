# %%
import pandas as pd
from sklearn.metrics import accuracy_score
from helper_classifier import set_seed

import os
import torch
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from helper_classifier import ClassifierModel


set_seed(42)

# %%
import platform
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()

# %%
def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

device = set_device()

# %% [markdown]
# ### Metadata

# %%
target_feature = "category"

# %% [markdown]
# ### Latents

# %%
def load_latents(target_feature='category'):
    df = pd.read_json(f"{DATA_PATH}/Zalando_Germany_Dataset/dresses/metadata/dresses_metadata.json").T.reset_index().rename(columns={'index': 'sku'})
    df = df[['sku', target_feature]].copy()

    # Drop rows with missing values
    df = df[df[target_feature].notna()]

    latents = torch.load(f"{DATA_PATH}/Models/e4e/experiments_default_lr/inversions/latents.pt").to(device)
    file_paths = pickle.load(open(f"{DATA_PATH}/Models/e4e/experiments_default_lr/inversions/file_paths.pkl", 'rb'))
    sku_ordering = [elem.split('/')[-1].split('.')[0] for elem in file_paths]
    df['latent_idx'] = df.sku.map(lambda x: sku_ordering.index(x) if x in sku_ordering else None)

    return df, latents

# %% [markdown]
# ### Generator

# %%
def setup_generator():
    os.chdir(f"{ROOT_PATH}/stylegan2-ada-pytorch")
    # Load model architecture
    experiment_path = f"{DATA_PATH}/Models/Stylegan2_Ada/Experiments/00003-stylegan2_ada_images-mirror-auto2-kimg1000-resumeffhq512/"
    model_name = "network-snapshot-000920.pkl"
    model_path = experiment_path + model_name
    with open(model_path, 'rb') as f:
        architecture = pickle.load(f)
        G = architecture['G_ema']
        D = architecture['D']
    G = G.to(device)
    os.chdir(current_wd)
    return G

# %% [markdown]
# ### DinoV2 Model

# %%
def setup_dinov2():
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    dino_device = 'cuda' if torch.cuda.is_available() else 'cpu' #Model does not work on MPS, therefore, only assign cuda, else CPU
    model = model.to(dino_device)

    return processor, model

# %%
def get_dinov2_embedding(model, processor, img:Image):
    inputs = processor(images=img, return_tensors="pt")
    dino_device = 'cuda' if torch.cuda.is_available() else 'cpu'  #Model does not work on MPS, therefore, only assign cuda, else CPU
    inputs = inputs.to(dino_device)
    with torch.no_grad():
        output = model(**inputs)
        embedding = output['pooler_output']
    return embedding

# %% [markdown]
# ### Load Attribute Classifier

# %%
def load_classifier():
    classifier = torch.load(f"{DATA_PATH}/Models/Assessor/DinoV2/Classifier/dinov2_category_classifier.pt")
    classifier = classifier.to(device)
    return classifier

# %% [markdown]
# ### Whole Attribute Score Pipeline

# %%
def get_attribute_scores(dino_model, dino_processor, classifier, img:Image):
    embedding = get_dinov2_embedding(dino_model, dino_processor, img)
    embedding = embedding.to(device)
    scores = classifier(embedding)
    scores = torch.softmax(scores, dim=1).squeeze()
    return scores


