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
from helper_DinoV2_Embeddings import *


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

dino_device, sg2_device, device = set_device()

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

    latents = torch.load(f"{DATA_PATH}/Models/e4e/experiments_default_lr/inversions/latents.pt")
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
    os.chdir(current_wd)
    return G

# %% [markdown]
# ### DinoV2 Model

# %%
def setup_dinov2():
    model_name = "facebook/dinov2-base"
    processor = dino_processor
    model = AutoModel.from_pretrained(model_name)

    return processor, model

# %%
def get_dinov2_embedding(model, processor, img):
    inputs = processor(img)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output = model(inputs)
        embedding = output['pooler_output']
    return embedding

# %% [markdown]
# ### Load Attribute Classifier

# %%
def load_classifier():
    classifier = torch.load(f"{DATA_PATH}/Models/Assessor/DinoV2/Classifier/dinov2_category_classifier.pt")
    return classifier

# %% [markdown]
# ### Whole Attribute Score Pipeline

# %%
def get_attribute_scores(dino_model, dino_processor, classifier, img):
    embedding = get_dinov2_embedding(dino_model, dino_processor, img)
    embedding = embedding.to(device)
    scores = classifier(embedding)
    scores = torch.softmax(scores, dim=1).squeeze()
    return scores


