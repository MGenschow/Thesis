from PIL import Image
import os
import pickle
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import InferenceData
from model import create_model
from labels import Labels
from train import combine_batches


def infer(
        checkpoint: str,
        img_dir: str,
        save_dir: str,
        batch_size: int = 256,
        N: int = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose: bool = True,
    ) -> None:

    model_id = checkpoint.split('/')[-1].split('-')[0]

    # DATA --------------------------------------------------------------------
    data = InferenceData(
        img_dir=img_dir,
        square=True,
        embedding_model='dinov2',
        N=N
    )
    iterator = DataLoader(data, batch_size=batch_size, shuffle=False)
    if verbose:
        print(f'Created dataset and dataloader. Number of images: {len(data)}\n')

    # MODEL -------------------------------------------------------------------

    # Load model info
    with open('model_ids.json', 'rb') as f:
        model_ids = json.load(f)
    model_info = model_ids[str(model_id)]

    # Embedding model
    dinov2_version = model_info["embeddings_name"]
    embedding_model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{dinov2_version}')
    embedding_model.eval()
    embedding_model.to(device)

    # Disentangling model
    labels = Labels(dataset_name=model_info['dataset_name'])
    disentangling_model = create_model(
        input_dim=model_info['input_dim'],
        hidden_dims_common=model_info['hidden_dims_common'],
        hidden_dims_branches=model_info['hidden_dims_branches'],
        output_dims=labels.task_label_dims,
        checkpoint=checkpoint,
        verbose=verbose
    ).to(device)
    disentangling_model.eval()

    # INFERENCE ---------------------------------------------------------------

    all_embeddings = []
    all_disentangled_embeddings = []
    all_predicted_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for filenames, images in tqdm(iterator, desc='Inference (batches)'):

            # Compute embeddings (output of model forward pass)
            embeddings = embedding_model(images.to(device))

            # Disentangle embeddings
            latent, output = disentangling_model(embeddings)

            # Get predicted labels
            predicted_labels = [head[i].argmax(dim=-1) for i, head in enumerate(output)]
            
            # Store
            all_embeddings.append(embeddings.cpu())
            all_predicted_labels.append(predicted_labels)
            all_disentangled_embeddings.append(latent)
            all_filenames += filenames

    all_embeddings = torch.cat(all_embeddings)
    all_disentangled_embeddings = combine_batches(all_disentangled_embeddings)
    all_predicted_labels = combine_batches(all_predicted_labels)

    # Replace task idx with task name (dict keys)
    def replace_idx_with_name(d: dict) -> dict:
        return {list(labels.task_mapping.keys())[k]: v for k, v in d.items()}
    all_disentangled_embeddings = replace_idx_with_name(all_disentangled_embeddings)
    all_predicted_labels = replace_idx_with_name(all_predicted_labels)

    # Replace predicted label index with label name
    idx_to_name = {k: {idx: name for name, idx in v.items()} for k, v in labels.task_mapping.items()}
    all_predicted_labels = {
        k: [idx_to_name[k][idx] for idx in v.numpy()]
        for k, v in all_predicted_labels.items()
    }

    # Save embeddings
    results = {
        'filenames': all_filenames,
        f'dinov2_{dinov2_version}': all_embeddings,
        'disentangled_embeddings': all_disentangled_embeddings,
        'predicted_labels': all_predicted_labels
    }
    os.makedirs(save_dir, exist_ok=True)
    save_file = f'{save_dir}/{model_id}.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(results, f)

    if verbose:
        print(f'Saved embeddings to {save_file}')


if __name__ == '__main__':

    model_id = 100
    data_dir = '../data/kastner-oehler'

    infer(
        checkpoint=f'models/{model_id}-model-best_valid_loss.pt',
        img_dir=f'{data_dir}/images',
        save_dir=f'{data_dir}/disentangled_embeddings',
        batch_size=64,
        verbose=True,
        N=None
    )
