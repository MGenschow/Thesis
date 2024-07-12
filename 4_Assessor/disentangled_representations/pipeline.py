import torch

from data import Embeddings
from train import train
from validate import train_validation_models

import platform
import os
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()


def train_and_validate(
        dataset_name: str = 'zalando_combined',
        embeddings_name: str = 'vits14',
        hidden_dims_common: list[int] = [256, 256],
        hidden_dims_branches: list[int] = [128, 128, 32],  # Last layer are the latents
        grl_weight: float = None,
        max_epochs: int = 100,
        use_early_stopping: bool = True,
        lr: float = 1e-3,
        ignore_missing_class: bool = True,
        batch_size: int = 256,
        prediction_loss_factor: float = 1,
        dcor_loss_factor: float = 0.,
        seed: int = 4243,
        checkpoint: str = None,
        device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        comment: str = None,
        verbose: bool = True
    ) -> float:
    """Train and validate a model."""

    if verbose:
        print(f'\nLOG: Using {device} device.')

    # Get embeddings
    embedding_args = dict(dataset_name=dataset_name, model_name=embeddings_name)
    train_data = Embeddings(**embedding_args, sample='train')
    valid_data = Embeddings(**embedding_args, sample='val')

    if verbose:
        print('\nLOG: Loaded data.\n')

    # Train the model
    model_id = train(
        train_data=train_data, 
        valid_data=valid_data,
        hidden_dims_common=hidden_dims_common,
        hidden_dims_branches=hidden_dims_branches,
        grl_weight=grl_weight,
        max_epochs=max_epochs,
        use_early_stopping=use_early_stopping,
        lr=lr,
        ignore_missing_class=ignore_missing_class,
        batch_size=batch_size,
        prediction_loss_factor=prediction_loss_factor,
        dcor_loss_factor=dcor_loss_factor,
        device=device,
        seed=seed,
        checkpoint=checkpoint,
        models_dir=f'{DATA_PATH}/Models/disentangled_representations/models/',
        dataset_name=dataset_name,
        embeddings_name=embeddings_name,
        comment=comment
    )

    if verbose:
        print('\nLOG: Trained model.\n')

    # Train validation model
    disentangling_metric = train_validation_models(
        model_id=model_id,
        select_model_by='best_valid_loss',
        hidden_dim=64,
        max_epochs=1000, 
        lr=1e-3,
        ignore_missing_class=ignore_missing_class,
        eval_metric='f1',
        device=device,
        seed=4243,
        verbose=True,
        models_dir=f'{DATA_PATH}/Models/disentangled_representations/validation_models/'
    )

    if verbose:
        print('\nLOG: Trained evaluation models.\n')

    return disentangling_metric


if __name__ == '__main__':

    import optparse

    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--dataset', action='store', type='string',
                         dest='dataset_name', default='zalando_combined',
                         help=('Dataset name (default %default)'))
    optParser.add_option('-e', '--embeddings', action='store', type='string',
                         dest='embeddings_name', default='vits14',
                         help=('DINOv2 model version (default %default)'))
    optParser.add_option('-g', '--grlweight', action='store', type='float',
                         dest='grl_weight', default=None,
                         help='Gradient reversal weight (default %default)')
    optParser.add_option('-c', '--dcor', action='store', type='float',
                         dest='dcor_loss_factor', default=0,
                         help=('Factor to scale the dcor loss(default %default)'))
    optParser.add_option('-m', '--comment', action='store', type='string',
                         dest='comment', default=None,
                         help=('Comment on the hyperparameters used '
                               '(default %default)'))
    
    opts, args = optParser.parse_args()

    train_and_validate(
        dataset_name=opts.dataset_name,
        embeddings_name=opts.embeddings_name,
        grl_weight=opts.grl_weight,
        dcor_loss_factor=opts.dcor_loss_factor,
        comment=opts.comment,
        verbose=True
    )
