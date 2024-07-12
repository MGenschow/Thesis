import torch

def distance_matrix(X: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Compute the distances along the last second dimension (feature dim)."""
    d = torch.norm(X[:, None, :] - X, dim=2, p=p)
    d_normalized = d - d.mean(dim=0, keepdim=True) - d.mean(dim=1, keepdim=True) + d.mean()
    return d_normalized

def distance_covariance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute the empirical distance covariance of two distance matrices."""
    return (X * Y).sum() / float(X.shape[0]**2)

def distance_correlation(X: torch.Tensor, Y: torch.Tensor):
    """
    Compute the distance correlation between two 2d-tensors.
    Tensors have shape (batch_dim, feature_dim)
    """
    A = distance_matrix(X)
    B = distance_matrix(Y)

    numerator = distance_covariance(A, B)
    denominator = torch.sqrt(distance_covariance(A, A) * distance_covariance(B, B))
    dcor = torch.sqrt(numerator/denominator)

    return dcor
