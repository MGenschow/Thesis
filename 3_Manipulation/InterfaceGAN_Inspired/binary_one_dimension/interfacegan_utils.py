###### Code adapted from https://github.com/genforce/interfacegan/tree/master



import os
import sys
import numpy as np
from sklearn import svm


def train_svm(latent_codes, labels, split_ratio=0.7):
    """Trains an SVM on latent codes with binary labels.
    
    Args:
        latent_codes: Input latent codes as training data.
        labels: Binary labels (0 or 1) shaped as [num_samples, 1].
        split_ratio: Ratio to split training and validation sets. (default: 0.7)
    
    Returns:
        A normalized decision boundary as `numpy.ndarray`.
    """
    
    # Input validation
    if not isinstance(latent_codes, np.ndarray) or latent_codes.ndim != 2:
        raise ValueError('`latent_codes` must be a 2D numpy.ndarray!')
    if not isinstance(labels, np.ndarray) or labels.shape != (latent_codes.shape[0], 1):
        raise ValueError('`labels` must be a numpy.ndarray with shape [num_samples, 1]!')

    # Flatten labels for compatibility with sklearn functions
    labels = labels.ravel()
    
    # Data splitting
    num_samples = len(latent_codes)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_point = int(num_samples * split_ratio)
    train_idx, val_idx = indices[:split_point], indices[split_point:]
    
    train_data, train_labels = latent_codes[train_idx], labels[train_idx]
    val_data, val_labels = latent_codes[val_idx], labels[val_idx]


    # Training the SVM
    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_labels)
    
    # Validation (if applicable)
    if len(val_data) > 0:
        val_predictions = classifier.predict(val_data)
        accuracy = np.mean(val_labels == val_predictions)
    
    # Normalize and return the decision boundary
    decision_boundary = classifier.coef_.reshape(1, -1).astype(np.float32)
    return decision_boundary / np.linalg.norm(decision_boundary), accuracy

def calculate_boundary(out_dir, out_name, latent_codes_path, scores_path):
    if not os.path.isfile(latent_codes_path):
        raise ValueError(f'Latent codes `{latent_codes_path}` does not exist!')
    latent_codes = np.load(latent_codes_path)

    if not os.path.isfile(scores_path):
        raise ValueError(f'Attribute scores `{scores_path}` does not exist!')
    scores = np.load(scores_path)

    boundary, accuracy = train_svm(latent_codes=latent_codes, labels=scores)
    
    np.save(os.path.join(out_dir, out_name), boundary)
    return accuracy