import torch
import torch.backends
from tqdm import tqdm
import os
import pickle
from torch.utils.data import DataLoader
from data import get_data
from train import combine_batches


import platform
import os
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()

def compute_embeddings(
        dataset_name: str,
        model_name: str = 'vits14',
        sample: str = 'train',
        batch_size: int = 256,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose: bool = True,
        **kwargs  # Further arguments for get_data
    ) -> None:
    """Compute the embeddings from DINOv2."""

    # Load model and move to device
    model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_name}')
    model.eval()
    model.to(device)

    # Get Dataset and create dataloader
    dataset = get_data(
        dataset_name=dataset_name,
        sample=sample,
        verbose=verbose,
        model='dinov2',
        **kwargs
    )
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(iterator, desc='Computing embeddings'):

            # Compute embeddings (output of model forward pass)
            embeddings = model(images.to(device))

            # Store in lists
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = combine_batches(all_labels)

    # Save embeddings
    results = {
        'embeddings': all_embeddings,
        'labels': all_labels,
        'images_ids': dataset.image_ids
    }
    embeddings_dir = f'{DATA_PATH}/Models/disentangled_representations/{dataset_name}/embeddings/{model_name}'
    os.makedirs(embeddings_dir, exist_ok=True)
    with open(f'{embeddings_dir}/{sample}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':

    # Command line arguments
    import optparse
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--dataset', action='store', type='string',
                         dest='dataset_name', default='zalando_combined',
                         help=('Dataset name (default %default)'))
    optParser.add_option('-m', '--model', action='store', type='string',
                         dest='model_name', default='vits14',
                         help=('DINOv2 model version (default %default)'))
    optParser.add_option('-s', '--sample', action='store', type='string',
                         dest='sample', default='train',
                         help=('Dataset sample (default %default)'))
    
    opts, args = optParser.parse_args()
    dataset_name = opts.dataset_name
    model_name = opts.model_name
    sample = opts.sample

    # Compute embeddings
    compute_embeddings(
        dataset_name=dataset_name,
        model_name=model_name,
        sample=sample,
        **({'img_type': 'square'} if dataset_name.startswith('zalando') else {})
    )    
