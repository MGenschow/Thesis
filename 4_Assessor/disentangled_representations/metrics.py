import torch
import torch.nn.functional as F

def compute_metric(
        preds: torch.tensor,
        targets: torch.tensor,
        metric: str,
        num_classes: int = None,
        ignore_missing_class: bool = True
    ) -> float:
    """
    Compute the metric for the given predictions and targets.

    Args:
    -----
    preds: torch.tensor
        Tensor of predictions.
    targets: torch.tensor
        Tensor of targets.
    metric: str
        The metric to compute.
    ignore_missing_class: bool
        Whether to ignore the 'Missing' class (target index 0).
    
    Returns:
    --------
    The computed metric.
    """
    if metric == 'accuracy':
        return preds.eq(targets.view_as(preds)).mean().item()
    elif metric == 'f1':

        # Convert preds and targets to one-hot
        preds = F.one_hot(preds.view_as(targets), num_classes=num_classes)
        targets = F.one_hot(targets, num_classes=num_classes)

        # Compute f1 score
        tp = (preds * targets).sum(dim=0).to(torch.float32)
        precision = tp / (preds.sum(dim=0).to(torch.float32) + 1e-8)
        recall = tp / (targets.sum(dim=0).to(torch.float32) + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        if ignore_missing_class:
            f1 = f1[1:] # Ignore the first class (0: Missing)
        return f1.mean().item()
    else:
        raise ValueError(f'Unknown metric {metric}')
