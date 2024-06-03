
import torch
import pickle
import platform
import os
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()


pti_base_path = f"{DATA_PATH}/Models/PTI/experiments/"
pti_models_path = f"{pti_base_path}checkpoints/"
pti_embeddings_path = f"{pti_base_path}embeddings/zalando_germany/PTI/"

def set_device():
    try:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    except:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    print(f"Using {device} as device")
    return device

device = set_device()



def load_pti(sku):
    os.chdir(f'{ROOT_PATH}/stylegan2-ada-pytorch/')
    # Get correct generator:
    G_PTI = torch.load(f"{pti_models_path}model_whole_sample_{sku}.pt", map_location=torch.device('cpu'))
    G_PTI = G_PTI.to(device)

    # Load corresponding latent embedding
    latent_path = f"{pti_embeddings_path}{sku}/0.pt"
    latent = torch.load(latent_path, map_location=torch.device('cpu'))
    latent = latent.to(device)

    os.chdir(current_wd)
    
    return G_PTI, latent