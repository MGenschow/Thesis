import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
import numpy as np
import time
import os
import pickle
from itertools import combinations

from model import create_model
from helpers import ModelIdStorage
from dcor import distance_correlation


import platform
import os
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()


def compute_loss_and_count_correct(
        latent: list[torch.Tensor],
        output: list[torch.Tensor],
        targets: list[torch.Tensor],
        criterion,
        prediction_loss_factor: float,
        dcor_loss_factor: float,
        grl_weight: float,
        device: str) -> tuple:
    """Compute the loss and number of correct predictions."""

    # Retrieve number of tasks
    n_tasks = len(output)
    drilldown_shape = (n_tasks, n_tasks)

    # Count corrects ----------------------------------------------------------
    corrects = np.zeros(drilldown_shape)
    for task1 in range(n_tasks):
        for task2 in range(n_tasks):
            targets_task = targets[task2].to(device)
            output_task = output[task1][task2]
            preds_task = output_task.argmax(-1, keepdim=True)
            corrects[task1, task2] = preds_task.eq(targets_task.view_as(preds_task)).sum().item()

    # Compute distance correlation loss ---------------------------------------

    # Loop through taskbranch combinations and compute distance correlation
    dcors = torch.ones(drilldown_shape)
    branch_combinations = list(combinations(range(n_tasks), r=2))
    for (task1, task2) in branch_combinations:
        X, Y = latent[task1], latent[task2]
        dcor = distance_correlation(X, Y)
        dcors[task1, task2] = dcor
        dcors[task2, task1] = dcor

    # Compute overall distance correlation loss (mean of lower triangualr matrix)
    dcor_loss = dcors[np.tril_indices(n_tasks, k=-1)].mean()
    dcor_loss *= dcor_loss_factor

    # Compute prediction loss -------------------------------------------------
    prediction_losses = torch.zeros(drilldown_shape)
    for task1 in range(n_tasks):
        for task2 in range(n_tasks):
            targets_task = targets[task2].to(device)
            output_task = output[task1][task2]
            prediction_losses[task1, task2] = criterion(output_task, targets_task)

    # Overall prediction loss (for backprop and stopping separately)
    prediction_loss_stopping = torch.diag(prediction_losses).mean() 
    prediction_loss_backprop = (
        prediction_losses.mean() if grl_weight is not None 
        else prediction_loss_stopping
    ) * prediction_loss_factor
    prediction_loss_stopping = prediction_loss_stopping * prediction_loss_factor

    # Compute total loss (prediction loss + dcor loss) ------------------------
    backprop_loss = prediction_loss_backprop + dcor_loss
    stopping_loss = (prediction_loss_stopping + dcor_loss).item()

    # Assemble loss drilldown -------------------------------------------------
    loss_drilldown = (
        torch.stack([dcors, prediction_losses], axis=0)
        .detach().cpu().numpy()
    )

    return corrects, backprop_loss, stopping_loss, loss_drilldown


def combine_batches(l: list) -> dict:
    """
    Convert a list of lists of tensors to a dict of tensors to combine batches
    whilst keeping tasks separated.
    """
    n_tasks = len(l[0])
    return {task_idx: torch.cat([batch[task_idx] for batch in l]).detach().cpu() 
            for task_idx in range(n_tasks)}


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        """
        Early stopping if validation loss does not decrease anymore.
        Source: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def train(
    train_data: torch.utils.data.Dataset,
    valid_data: torch.utils.data.Dataset,
    hidden_dims_common = list[int],
    hidden_dims_branches = list[int],
    grl_weight: float = None,
    max_epochs: int = 100,
    use_early_stopping: bool = True,
    lr: float = 5e-4,
    batch_size: int = 128,
    ignore_missing_class: bool = True,  # Ignore missing class in loss computation
    prediction_loss_factor: float = 1,
    dcor_loss_factor: float = 0,
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    seed: int = 4243,
    checkpoint: str = None,
    verbose: bool = True,
    models_dir: str = f'{DATA_PATH}/Models/disentangled_representations/models/',
    dataset_name: str = None,
    embeddings_name: str = None,
    comment: str = ''
) -> float:
    """Train a model, i.e. loop through epochs to train and evaluate."""

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # DATA --------------------------------------------------------------------

    # Create data loaders
    train_iterator = data.DataLoader(
        train_data, shuffle=True, batch_size=batch_size
    )
    valid_iterator = data.DataLoader(valid_data, batch_size=batch_size)
    input_dim = train_data[0][0].shape[0]

    # MODEL -------------------------------------------------------------------

    # Create new model_id and add to storage
    model_id_storage = ModelIdStorage()
    model_id = model_id_storage.new_id()

    model_hyperparams = {
        'comment': comment,
        'input_dim': input_dim,
        'hidden_dims_common': hidden_dims_common,
        'hidden_dims_branches': hidden_dims_branches,
        'grl_weight': grl_weight,
        'max_epochs': max_epochs,
        'lr': lr,
        'batch_size': batch_size,
        'prediction_loss_factor': prediction_loss_factor,
        'dcor_loss_factor': dcor_loss_factor,
        'seed': seed,
        'checkpoint': checkpoint,
        'dataset_name': dataset_name,
        'embeddings_name': embeddings_name,
    }
    model_id_storage.store_info(model_hyperparams)

    if verbose:
        print(f'Creating new model with ID: {model_id}')

    # Create model and move to device
    output_dims = train_data.Labels.task_label_dims
    model = create_model(
        input_dim=input_dim,
        hidden_dims_common=hidden_dims_common,
        hidden_dims_branches=hidden_dims_branches,
        output_dims=output_dims,
        grl_weight=grl_weight,
        checkpoint=None if checkpoint is None else models_dir+checkpoint+'.pt',
        verbose=verbose
    ).to(device)

    # Create directory for model checkpoints and statistics
    os.makedirs(models_dir, exist_ok=True)
    
    # OPTIMIZER ---------------------------------------------------------------

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # LOSS --------------------------------------------------------------------

    # Define the loss function
    criterion_args = dict(ignore_index=0) if ignore_missing_class else {}
    criterion = nn.CrossEntropyLoss(**criterion_args)

    # Move criterion to device
    criterion = criterion.to(device)

    # Assemble loss args (same for train and validation loss)
    loss_args = dict(
        criterion=criterion,
        prediction_loss_factor=prediction_loss_factor,
        dcor_loss_factor=dcor_loss_factor,
        grl_weight=grl_weight,
        device=device
    )

    # LOOP --------------------------------------------------------------------

    n_tasks = len(train_data.Labels.task_label_dims)

    # Empty lists to store metrics in
    train_losses = []
    valid_losses = []
    train_losses_drilldown = []
    valid_losses_drilldown = []
    train_accuracies = []
    valid_accuracies = []
    train_accuracies_drilldown = []
    valid_accuracies_drilldown = []
    elapsed_time = []

    # Init variables to select best model
    best_valid_loss = np.Inf
    best_valid_acc = 0

    # Init early stopper
    if use_early_stopping:
        early_stopper = EarlyStopper(patience=max_epochs/10, min_delta=0)

    # Loop through epochs
    start_time = time.time()
    for epoch in trange(max_epochs, desc='Epochs'):

        # Initialize storage for data that is needed for the validation model
        train_latent_reps = []
        train_preds = []
        train_reals = []
        valid_latent_reps = []
        valid_preds = []
        valid_reals = []
        
        # Initialize running variables (for aggregation across batches)
        train_loss = 0
        valid_loss = 0
        loss_drilldown_shape = (2, len(output_dims), len(output_dims))
        train_loss_drilldown = np.zeros(loss_drilldown_shape)
        valid_loss_drilldown = np.zeros(loss_drilldown_shape)
        corrects_shape = (n_tasks, n_tasks)
        train_correct = np.zeros(corrects_shape)
        valid_correct = np.zeros(corrects_shape)
        
        # === Training === #
        model.train()
        
        # Loop through train batches
        for batch, targets in tqdm(train_iterator, desc='Training', leave=False):

            # Move batch to device and make forward pass
            batch = batch.to(device)
            latent, output = model(batch)

            # Compute loss and count corrects
            corrects, backprop_loss, stopping_loss, loss_drilldown = compute_loss_and_count_correct(
                latent=latent, output=output, targets=targets, **loss_args
            )
            
            # Backward pass (only if not first epoch)
            if epoch > 0:
                optimizer.zero_grad()
                backprop_loss.backward()
                optimizer.step()
            
            # Update running loss and corrects variable
            train_correct += corrects
            train_loss += stopping_loss
            train_loss_drilldown += loss_drilldown

            # Store data for validation model
            train_latent_reps.append(latent)
            train_preds.append([out[i] for i, out in enumerate(output)]) #FIXME
            train_reals.append(targets)
        
        # Compute epoch loss and accuracy on training set
        train_accuracies.append(np.mean(np.diag(train_correct)/len(train_data)))
        train_accuracies_drilldown.append(train_correct/len(train_data))
        train_losses.append(train_loss/len(train_iterator))
        train_losses_drilldown.append(train_loss_drilldown/len(train_iterator))

        # === Evaluation on validation sample === #
        model.eval()

        # Loop through validation batches
        for batch, targets in tqdm(valid_iterator, desc='Evaluation', leave=False):

            # Move batch to device and make forward pass
            batch = batch.to(device)
            latent, output = model(batch)

            # Compute loss and count corrects
            corrects, backprop_loss, stopping_loss, loss_drilldown = compute_loss_and_count_correct(
                latent=latent, output=output, targets=targets, **loss_args
            )
            
            # Update running loss variable
            valid_correct += corrects
            valid_loss += stopping_loss
            valid_loss_drilldown += loss_drilldown

            # Store data for validation model
            valid_latent_reps.append(latent)
            valid_preds.append([out[i] for i, out in enumerate(output)]) #FIXME
            valid_reals.append(targets)
        
        # Compute epoch loss and accuracy on validation set
        valid_acc = np.mean(np.diag(valid_correct)/len(valid_data))
        valid_accuracies.append(valid_acc)
        valid_accuracies_drilldown.append(valid_correct/len(valid_data))
        valid_loss = valid_loss/len(valid_iterator)
        valid_losses.append(valid_loss)
        valid_losses_drilldown.append(valid_loss_drilldown/len(valid_iterator))
        
        # Save elpased time of training until current epoch
        time_till_epoch = time.time() - start_time
        elapsed_time.append(time_till_epoch)

        def save_validation_data(filename_end: str) -> None:
            """Save the data needed for the validation model"""
            d = {
                'train': {
                    'latent_reps': train_latent_reps,
                    'reals': train_reals, 
                    'preds': train_preds
                },
                'val': {
                    'latent_reps': valid_latent_reps,
                    'reals': valid_reals, 
                    'preds': valid_preds
                }
            }

            # Combine (concatenate) batches
            d = {
                k0: {k1: combine_batches(v1) for k1, v1 in v0.items()} 
                for k0, v0 in d.items()
            }

            with open(f'{models_dir}{model_id}-latent_reps-{filename_end}.pkl', 'wb') as f:
                pickle.dump(d, f)
        

        # Save model and data needed for the validation model
        # By best accuracy
        if valid_acc > best_valid_acc:
            torch.save(model.state_dict(), f'{models_dir}{model_id}-model-best_valid_acc.pt')
            best_epoch_by_valid_acc = epoch
            time_till_best_epoch_by_valid_acc = time_till_epoch
            best_valid_acc = valid_acc

            save_validation_data(filename_end='best_valid_acc')

        # By best loss
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), f'{models_dir}{model_id}-model-best_valid_loss.pt')
            best_epoch_by_valid_loss = epoch
            time_till_best_epoch_by_valid_loss = time_till_epoch
            best_valid_loss = valid_loss

            save_validation_data(filename_end='best_valid_loss')

        # By last epoch
        if epoch == max_epochs-1:
            torch.save(model.state_dict(), f'{models_dir}{model_id}-model-last_epoch.pt')
            save_validation_data(filename_end='last_epoch')

        # Assemble metrics in dictionary and save to disk
        results = {
            'model_id': model_id,
            
            'accuracy': {
                'train': train_accuracies, 
                'validation': valid_accuracies, 
                'best': best_valid_acc,
                'best_epoch': best_epoch_by_valid_acc,
                'time_till_best_epoch': time_till_best_epoch_by_valid_acc,
            }, 
            
            'loss': {
                'train': train_losses, 
                'validation': valid_losses, 
                'best': best_valid_loss,
                'best_epoch': best_epoch_by_valid_loss,
                'time_till_best_epoch': time_till_best_epoch_by_valid_loss,
            },
            
            'loss_drilldown': {
                'train': train_losses_drilldown,
                'validation': valid_losses_drilldown
            },

            'accuracy_drilldown': {
                'train': train_accuracies_drilldown,
                'validation': valid_accuracies_drilldown
            },

            'elapsed_time': elapsed_time,

            'dataset_name': dataset_name,
        }

        with open(f'{models_dir}{model_id}-training_stats.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Check early stopping criterium
        if use_early_stopping:
            if early_stopper.early_stop(valid_loss):
                if verbose:
                    print(f'\nStopping after {epoch} epochs.')             
                break

    # RESULTS -----------------------------------------------------------------

    # Print best model's statistics
    if verbose:
        print('\n')
        print(
            f'Best model (by loss on validation set):\n'
            f'\tAccuracy={round(valid_accuracies[best_epoch_by_valid_loss].mean(), 4)}, '
            f'Loss={round(valid_losses[best_epoch_by_valid_loss], 4)}, '
            f'after {best_epoch_by_valid_loss+1} epochs '
            f'({round(time_till_best_epoch_by_valid_loss, 2)}s)'
            
            
            f'\nBest model (by accuracy on validation set):\n'
            f'\tAccuracy={round(valid_accuracies[best_epoch_by_valid_acc].mean(), 4)}, '
            f'Loss={round(valid_losses[best_epoch_by_valid_acc], 4)}, '
            f'after {best_epoch_by_valid_acc+1} epochs '
            f'({round(time_till_best_epoch_by_valid_acc, 2)}s)'
        )

    return model_id
