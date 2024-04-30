# %%
import torch
from transformers import AutoImageProcessor, AutoModel
from glob import glob
from PIL import Image
from tqdm.notebook import tqdm
import numpy as np
import os
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import torchvision.transforms as transforms

# %%
import platform
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

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

def set_dino_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using dino device: {device}")
    return device
# %%
# Setup DinoV2 Custom Processor to ensure gradient flow in later training
transform_pipeline = transforms.Compose([
    #transforms.Resize(256),  # Resize so the shortest side is 256
    #transforms.CenterCrop((224, 224)),  # Center crop to 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
def dino_processor(input):
    if isinstance(input, str):
        img = Image.open(input).convert('RGB')
        img = transforms.ToTensor()(img.resize([512,512]))
        img = img.unsqueeze(0)

        processed_img = transform_pipeline(img)
    elif isinstance(input, torch.Tensor):
        processed_img = transform_pipeline(input)
    else:
        raise ValueError("Input must be either a string or a torch.Tensor")
    return processed_img

# %%
# Extract all embeddings
def extract_embeddings():
    df = pd.read_json(f"{DATA_PATH}/Zalando_Germany_Dataset/dresses/metadata/dresses_metadata.json").T.reset_index().rename(columns={'index': 'sku'})
    save_path = f"{DATA_PATH}/Models/Assessor/DinoV2/Embeddings/dinov2_embeddings.pt"
    root_path = f"{DATA_PATH}/Zalando_Germany_Dataset/dresses/images/square_images/"

    if not os.path.exists(save_path):
        print('Calculating embeddings from DINOV2 model...')

        model_name = "facebook/dinov2-base"
        device = set_dino_device()
        processor = dino_processor
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)


        embeddings = torch.zeros(df.shape[0], 768)

        for row in tqdm(df.iterrows(), total=df.shape[0]):
            index = row[0]
            sku = row[1]['sku']
            # Load Image and preprocess
            img_path = f"{root_path}{sku}.jpg"
            input = processor(img_path)
            input = input.to(device)
            # Perform forward pass
            with torch.no_grad():
                output = model(input)
                embedding = output['pooler_output']
            # Assign embedding to embeddings
            embeddings[index,:] = embedding


        # Save embeddings to disc
        torch.save(embeddings, save_path)
        print('Embeddings saved to disk...')
    else: 
        print('Loading embeddings from disk...')
        embeddings = torch.load(save_path)
        print(f'{embeddings.shape[0]} embeddings loaded from disk...')
    
    return embeddings, df

# %%


# %%



