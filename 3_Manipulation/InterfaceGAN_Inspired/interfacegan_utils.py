import os
import sys
import logging
import numpy as np
from sklearn import svm

def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
  """Sets up logger from target work directory.

  The function will sets up a logger with `DEBUG` log level. Two handlers will
  be added to the logger automatically. One is the `sys.stdout` stream, with
  `INFO` log level, which will print improtant messages on the screen. The other
  is used to save all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will
  be added time stamp and log level before logged.

  NOTE: If `work_dir` or `logfile_name` is empty, the file stream will be
  skipped.

  Args:
    work_dir: The work directory. All intermediate files will be saved here.
      (default: None)
    logfile_name: Name of the file to save log message. (default: `log.txt`)
    logger_name: Unique name for the logger. (default: `logger`)

  Returns:
    A `logging.Logger` object.

  Raises:
    SystemExit: If the work directory has already existed, of the logger with
      specified name `logger_name` has already existed.
  """

  logger = logging.getLogger(logger_name)
  if logger.hasHandlers():  # Already existed
    raise SystemExit(f'Logger name `{logger_name}` has already been set up!\n'
                     f'Please use another name, or otherwise the messages '
                     f'may be mixed between these two loggers.')

  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

  # Print log message with `INFO` level or above onto the screen.
  sh = logging.StreamHandler(stream=sys.stdout)
  sh.setLevel(logging.INFO)
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  if not work_dir or not logfile_name:
    return logger

  if os.path.exists(work_dir):
    raise SystemExit(f'Work directory `{work_dir}` has already existed!\n'
                     f'Please specify another one.')
  os.makedirs(work_dir)

  # Save log message with all levels in log file.
  fh = logging.FileHandler(os.path.join(work_dir, logfile_name))
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  return logger



def train_boundary(latent_codes, labels, split_ratio=0.7, logger=None):
    """Trains an SVM on latent codes with binary labels.
    
    Args:
        latent_codes: Input latent codes as training data.
        labels: Binary labels (0 or 1) shaped as [num_samples, 1].
        split_ratio: Ratio to split training and validation sets. (default: 0.7)
        logger: Logger for recording log messages. If None, uses a default logger.
    
    Returns:
        A normalized decision boundary as `numpy.ndarray`.
    """
    if not logger:
        logger = setup_logger(work_dir='', logger_name='train_boundary')
    
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
    logger.info('Training SVM model...')
    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_labels)
    logger.info('Model training complete.')
    
    # Validation (if applicable)
    if len(val_data) > 0:
        val_predictions = classifier.predict(val_data)
        accuracy = np.mean(val_labels == val_predictions)
        logger.info(f'Validation accuracy: {accuracy:.2f}')
    
    # Normalize and return the decision boundary
    decision_boundary = classifier.coef_.reshape(1, -1).astype(np.float32)
    return decision_boundary / np.linalg.norm(decision_boundary)

