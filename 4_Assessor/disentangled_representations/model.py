import torch
import torch.nn as nn
from pytorch_adapt.layers.gradient_reversal import GradientReversal


def print_n_params(model, prefix: str = 'The'):
    """Print the number of trainable parameters of a model."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{prefix} model has {n_params:,} trainable parameters.')


class TaskBranches(nn.Module):
    def __init__(
            self, 
            input_dim: int,
            hidden_dims_common: list[int],
            hidden_dims_branches: list[int],
            output_dims: list,
            grl_weight: float = None
        ) -> None:
        """Neural network with a MLP branch for each task, e.g. length."""

        super().__init__()

        self.n_tasks = len(output_dims)
        self.activation = nn.ReLU()
        self.n_hidden_layers_common = len(hidden_dims_common)
        self.n_hidden_layers_branches = len(hidden_dims_branches)
        self.use_grl = (grl_weight is not None)
        if self.use_grl:
            self.grl = GradientReversal(weight=grl_weight)

        # Build common layers
        for i, hidden_dim in enumerate(hidden_dims_common):
            fc = nn.Linear(input_dim, hidden_dim)
            setattr(self, f'common_fc{i}', fc)
            input_dim = hidden_dim

        # Build sub-nets for each branch
        for branch in range(self.n_tasks):

            branch_input_dim = input_dim

            # Build branch layers
            for i, hidden_dim in enumerate(hidden_dims_branches):
                fc = nn.Linear(branch_input_dim, hidden_dim)
                setattr(self, f'branch_{branch}_fc{i}', fc)
                branch_input_dim = hidden_dim

            # Output layer
            for task in range(self.n_tasks):
                out = nn.Linear(branch_input_dim, output_dims[task])
                setattr(self, f'branch_{branch}_task_{task}_out', out)       

    def forward(self, x):

        # Get common layers outputs (post-activation)
        common_out = x
        for j in range(self.n_hidden_layers_common):
            common_out = self.activation(getattr(self, f'common_fc{j}')(common_out))

        # Get branch layers outputs (post-activation, only last layer pre-activation)
        latents = []
        for i in range(self.n_tasks):
            latent = common_out
            for j in range(self.n_hidden_layers_branches):
                latent = getattr(self, f'branch_{i}_fc{j}')(latent)
                if j+1 < self.n_hidden_layers_branches:
                    latent = self.activation(latent)
            latents.append(latent)

        # Output layer
        outs = []
        for branch, latent in enumerate(latents):

            # Apply activation to latent space
            latent = self.activation(latent)

            # Get all tasks outputs that are based on the branch's latent space
            out_branch = []
            for task in range(self.n_tasks):

                # Apply output layer
                out = getattr(self, f'branch_{branch}_task_{task}_out')(latent)

                # Apply GRL (only if not the primary task)
                if self.use_grl:
                    if (branch != task):
                        out = self.grl(out)

                out_branch.append(out)

            # Append to output list
            outs.append(out_branch)
        
        return latents, outs
    

class ValidationModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Simple MLP for validation"""
        super().__init__()

        # Define activation
        self.activation = nn.ReLU()

        # Define layers
        fc1 = nn.Linear(input_dim, hidden_dim)
        fc2 = nn.Linear(hidden_dim, hidden_dim)
        out = nn.Linear(hidden_dim, output_dim)

        # Set as attributes
        setattr(self, 'fc1', fc1)
        setattr(self, 'fc2', fc2)
        setattr(self, 'out', out)

        # Append to list
        self.layers = [fc1, fc2, out]

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # Do not apply activation on last layer
            if i+1 < len(self.layers):
                out = self.activation(out)
        return out
    

def create_model(
        input_dim: int,
        hidden_dims_common: list[int],
        hidden_dims_branches: list[int],
        output_dims: list[int],
        grl_weight: float = None,
        checkpoint: str = None,
        verbose: bool = False,
        device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
    """Create the neural network model."""

    # Network with a branch for each task
    model = TaskBranches(
        input_dim=input_dim,
        hidden_dims_common=hidden_dims_common,
        hidden_dims_branches=hidden_dims_branches,
        output_dims=output_dims,
        grl_weight=grl_weight
    )

    if verbose:
        print_n_params(model=model, prefix='The TaskBranches')

    # Print model architecture
    if verbose:
        print('\nModel architecture:')
        print(model)
        print()

    # Load parameters from checkpoint
    if checkpoint is not None:
        checkpoint_params = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_params)

        if verbose:
            print(f'Checkpoint loaded from {checkpoint}\n')

    return model
