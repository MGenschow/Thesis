
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
import numpy as np
import time
import os
import pickle

from train import EarlyStopper
from model import ValidationModel
from metrics import compute_metric


import platform
import os
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()


def train_validation_models(
        model_id: int,
        select_model_by: str = 'best_valid_acc',
        hidden_dim: int = 64,
        max_epochs: int = 1000, 
        lr: float = 1e-4,
        batch_size: int = 128,
        ignore_missing_class: bool = True,  # Ignore missing class in loss computation
        eval_metric: str = 'f1',
        device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        seed: int = 4243,
        verbose: bool = True,
        models_dir: str = f'{DATA_PATH}/Models/disentangled_representations/validation_models/'
    ):
    """Train validation model, i.e. simple MLP using the latent representations."""

    # Read data from file
    with open(f'{DATA_PATH}/Models/disentangled_representations/models/{model_id}-latent_reps-{select_model_by}.pkl', 'rb') as f:
        data = pickle.load(f)

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create directory for model checkpoints and statistics
    os.makedirs(models_dir, exist_ok=True)

    # Empty dictionary for validation metrics per task-combination
    metrics = {}

    # Empty dictionary for validation model predictions
    predictions_all_models = {}

    # Loop through tasks (all feature-target combinations)
    task_idxs = data['train']['latent_reps'].keys()
    for feat_task_idx in task_idxs:
        for target_task_idx in task_idxs:
        
            if verbose:
                print('\nValidation model '
                      f'using features from task {feat_task_idx} '
                      f'and targets from task {target_task_idx}')
                
            # DATA ------------------------------------------------------------

            # Create datasets
            train_data = TensorDataset(
                data['train']['latent_reps'][feat_task_idx], 
                data['train']['reals'][target_task_idx]
            )
            valid_data = TensorDataset(
                data['val']['latent_reps'][feat_task_idx], 
                data['val']['reals'][target_task_idx]
            )

            # Create data loaders
            train_iterator = DataLoader(
                train_data, shuffle=True, batch_size=batch_size
            )
            valid_iterator = DataLoader(valid_data, batch_size=batch_size)

            # MODEL -----------------------------------------------------------

            # Create model and move to device
            n_attributes = data['train']['preds'][target_task_idx].shape[1]
            model = ValidationModel(
                input_dim=data['train']['latent_reps'][feat_task_idx].shape[1],
                hidden_dim=hidden_dim,
                output_dim=n_attributes
            ).to(device)

            # OPTIMIZER -------------------------------------------------------

            optimizer = optim.Adam(model.parameters(), lr=lr)

            # LOSS ------------------------------------------------------------

            # Define the loss function
            criterion_args = dict(ignore_index=0) if ignore_missing_class else {}
            criterion = nn.CrossEntropyLoss(**criterion_args)
            criterion = criterion.to(device)

            # LOOP ------------------------------------------------------------

            # Empty lists to store metrics in
            train_losses = []
            train_metrics = []
            valid_losses = []
            valid_metrics = []
            elapsed_time = []

            # Empty lists to store outputs in
            predictions = {'train': [], 'validation': []}

            # Init variables to select best model
            best_valid_loss = np.Inf
            best_valid_metric = 0
            best_epoch_by_valid_metric = 0
            best_epoch_by_valid_loss = 0
            time_till_best_epoch_by_valid_metric = 0
            time_till_best_epoch_by_valid_loss = 0

            # Init early stopper
            early_stopper = EarlyStopper(patience=max_epochs/10, min_delta=0)

            # Loop through epochs
            start_time = time.time()
            for epoch in trange(max_epochs, desc='Epochs', leave=False):

                # Initialize running variables (for aggregation across batches)
                train_loss = 0
                train_metric = 0
                valid_loss = 0
                valid_metric = 0
                predictions_epoch = {'train': [], 'validation': []}
                
                # === Training === #
                model.train()

                # Loop through train batches
                for input_train, targets_train in train_iterator:
                
                    # Forward pass
                    input_train = input_train.to(device)
                    targets_train = targets_train.to(device)
                    output_train = model(input_train)

                    # Compute loss
                    loss = criterion(output_train, targets_train)
                
                    # Backward pass
                    if epoch > 0:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                            
                    # Make predictions
                    preds_train = output_train.detach().argmax(-1, keepdim=True)
                    predictions_epoch['train'].append(preds_train.cpu().numpy())
                
                    # Update running loss and metric on training set
                    train_metric += compute_metric(
                        preds=preds_train,
                        targets=targets_train,
                        metric=eval_metric,
                        num_classes=n_attributes,
                        ignore_missing_class=ignore_missing_class
                    )*len(targets_train)  # Weighted by batch size
                    train_loss += loss.item()

                # Compute epoch loss and accuracy on training set
                train_metrics.append((train_metric/len(train_data)))
                train_losses.append(train_loss/len(train_data))

                # === Evaluation on validation sample === #
                model.eval()

                # Loop through test batches
                for input_valid, targets_valid in valid_iterator:

                    # Forward pass
                    input_valid = input_valid.to(device)
                    targets_valid = targets_valid.to(device)
                    output_valid = model(input_valid)

                    # Compute loss
                    loss = criterion(output_valid, targets_valid)
                        
                    # Make predictions
                    preds_valid = output_valid.detach().argmax(-1, keepdim=True)
                    predictions_epoch['validation'].append(preds_valid.cpu().numpy())
                
                    # Update running loss and corrects on validation set
                    valid_metric += compute_metric(
                        preds=preds_valid,
                        targets=targets_valid,
                        metric=eval_metric,
                        num_classes=n_attributes,
                        ignore_missing_class=ignore_missing_class
                    )*len(targets_valid)  # Weighted by batch size
                    valid_loss += loss.item()
                
                # Compute epoch loss and accuracy on validation set
                valid_metric = valid_metric/len(valid_data)
                valid_loss = valid_loss/len(valid_data)
                valid_metrics.append(valid_metric)
                valid_losses.append(valid_loss)

                # Store predictions
                predictions['train'].append(
                    np.concatenate(predictions_epoch['train']).flatten()
                )
                predictions['validation'].append(
                    np.concatenate(predictions_epoch['validation']).flatten()
                )
                
                # Save elpased time of training until current epoch
                time_till_epoch = time.time() - start_time
                elapsed_time.append(time_till_epoch)
                
                # Save best model
                model_name = (
                    f'model_id={model_id}'
                    f'-feat_task={feat_task_idx}'
                    f'-target_task={target_task_idx}'
                    f'-max_epochs={max_epochs}'
                    f'-lr={lr}'
                    f'-seed={seed}'
                )

                if valid_metric > best_valid_metric:
                    torch.save(model.state_dict(), f'{models_dir}{model_name}-best_valid_metric.pt')
                    best_epoch_by_valid_metric = epoch
                    time_till_best_epoch_by_valid_metric = time_till_epoch
                    best_valid_metric = valid_metric
                    
                if valid_loss < best_valid_loss:
                    torch.save(model.state_dict(), f'{models_dir}{model_name}-best_valid_loss.pt')
                    best_epoch_by_valid_loss = epoch
                    time_till_best_epoch_by_valid_loss = time_till_epoch
                    best_valid_loss = valid_loss

                # Assemble metrics in dictionary and save to disk
                results = {
                    'model_name': model_name,
                    
                    'loss': {
                        'train': train_losses, 
                        'validation': valid_losses, 
                        'best': best_valid_loss,
                        'best_epoch': best_epoch_by_valid_loss,
                        'time_till_best_epoch': time_till_best_epoch_by_valid_loss,
                    },
                    
                    'eval_metric': {
                        'name': eval_metric,
                        'train': train_metrics, 
                        'validation': valid_metrics, 
                        'best': best_valid_metric,
                        'best_epoch': best_epoch_by_valid_metric,
                        'time_till_best_epoch': time_till_best_epoch_by_valid_metric,
                    },        
                    
                    'elapsed_time': elapsed_time,
                }

                with open(f'{models_dir}{model_name}-training_stats.pkl', 'wb') as f:
                    pickle.dump(results, f)

                # Check early stopping criterium
                if early_stopper.early_stop(valid_loss):
                    if verbose:
                        print(f'\nStopping after {epoch} epochs.')             
                    break

            # RESULTS ---------------------------------------------------------

            # Print best model's statistics
            if verbose:
                print(
                    f'Best model (by loss on validation set):\n'
                    f'\t{eval_metric}={round(valid_metrics[best_epoch_by_valid_loss], 4)}, '
                    f'Loss={round(valid_losses[best_epoch_by_valid_loss], 4)}, '
                    f'after {best_epoch_by_valid_loss+1} epochs '
                    f'({round(time_till_best_epoch_by_valid_loss, 2)}s)'
                    
                    
                    f'\nBest model (by {eval_metric} on validation set):\n'
                    f'\t{eval_metric}={round(valid_metrics[best_epoch_by_valid_metric], 4)}, '
                    f'Loss={round(valid_losses[best_epoch_by_valid_metric], 4)}, '
                    f'after {best_epoch_by_valid_metric+1} epochs '
                    f'({round(time_till_best_epoch_by_valid_metric, 2)}s)'
                )

            # Save predictions to dictionary
            predictions_all_models[(feat_task_idx, target_task_idx)] = {
                'train': np.stack(predictions['train']),
                'validation': np.stack(predictions['validation'])
            }

            # Compute baseline metric
            train_targets = train_data.tensors[1]
            if ignore_missing_class:
                train_targets = train_targets[train_targets != 0]
            most_common_class = train_targets.mode(dim=-1).values
            baseline_valid_metric = compute_metric(
                preds=torch.full_like(valid_data.tensors[1], most_common_class),
                targets=valid_data.tensors[1],
                metric=eval_metric,
                num_classes=n_attributes,
                ignore_missing_class=ignore_missing_class
            )

            # Save best validation metric value to dictionary
            metrics[(feat_task_idx, target_task_idx)] = {
                'best': best_valid_metric,
                'baseline': baseline_valid_metric
            }

    # AGGREGATE METRICS -------------------------------------------------------

    # Average improvement over baseline
    primary_tasks_baseline_improvement = np.mean([
        v['best'] / v['baseline'] - 1 for k, v in metrics.items() if k[0] == k[1]
    ])
    other_tasks_baseline_improvement = np.mean([
        v['best'] / v['baseline'] - 1 for k, v in metrics.items() if k[0] != k[1]
    ])
    baseline_improvement_factor = primary_tasks_baseline_improvement / other_tasks_baseline_improvement
    baseline_improvement_diff = primary_tasks_baseline_improvement - other_tasks_baseline_improvement

    # Average improvement of feature usability
    feature_factors = []
    for feat_task_idx in task_idxs:
        feature_factor = np.mean([
            metrics[(feat_task_idx, feat_task_idx)]['best'] / v['best']
            for k, v in metrics.items()
            if (k[0] == feat_task_idx) and (k[1] != feat_task_idx)
        ])
        feature_factors.append(feature_factor)
    avg_feature_factor = np.mean(feature_factors)

    # Harmonic mean
    agg_metrics = {
        'baseline_improvement_factor': baseline_improvement_factor,
        'baseline_improvement_diff': baseline_improvement_diff,
        'feature_usability_factor': avg_feature_factor
    }
    agg_metrics_values = np.array(list(agg_metrics.values()))
    disentangling_metric = len(agg_metrics_values) / (1/agg_metrics_values).sum()
    agg_metrics['harmonic_mean'] = disentangling_metric

    if verbose:
        print('\nAggregated metrics:')
        for k, v in agg_metrics.items():
            print(f'\t{k}: {v}')

    # SAVE TO DISK ------------------------------------------------------------

    # Save metrics to disk
    with open(f'{models_dir}{model_id}-metrics.pkl', 'wb') as f:
        d = {'metric': eval_metric, 'values': metrics, 'summary': agg_metrics}
        pickle.dump(d, f)

    # Save predictions to disk
    with open(f'{models_dir}{model_id}-predictions.pkl', 'wb') as f:
        pickle.dump(predictions_all_models, f)

    # Return summary metric for tuning
    return disentangling_metric
