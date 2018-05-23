import collections
import os.path

from . import training


work_path = '_workspace'
"""str: Path to parent directory containing program output."""

extraction_path = os.path.join(work_path, 'features')
"""str: Path to the directory containing extracted feature vectors."""

scaler_path = os.path.join(extraction_path, 'scaler.p')
"""str: Path to the scaler file used for standardization."""

model_path = os.path.join(work_path, 'models', training.training_id)
"""str: Path to the output directory of saved models."""

log_path = os.path.join(work_path, 'logs', training.training_id)
"""str: Path to the directory of TensorBoard logs."""

history_path = os.path.join(log_path, 'history.csv')
"""str: Path to log file for training history."""

predictions_path = os.path.join(
    work_path, 'predictions', training.training_id, '{}_{}_predictions.p')
"""str: Path to a model predictions file."""

results_path = os.path.join(
    work_path, 'results', training.training_id, '{}_{}_results.csv')
"""str: Path to the file containing results."""


Dataset = collections.namedtuple('Dataset',
                                 ['name',
                                  'path',
                                  'metadata_path',
                                  ])
"""Data structure encapsulating information about a dataset."""

_root_dataset_path = ('/vol/vssp/AP_datasets/audio/audioset/'
                      'task4_dcase2017_audio/official_downloads')
"""str: Path to root directory containing input audio clips."""

training_set = Dataset(
    name='training',
    path=os.path.join(_root_dataset_path, 'training'),
    metadata_path='metadata/groundtruth_weak_label_training_set.csv',
)
"""Dataset instance for the training dataset."""

validation_set = Dataset(
    name='validation',
    path=os.path.join(_root_dataset_path, 'testing'),
    metadata_path='metadata/groundtruth_strong_label_testing_set.csv',
)
"""Dataset instance for the validation dataset."""

test_set = Dataset(
    name='test',
    path=os.path.join(_root_dataset_path, 'evaluation'),
    metadata_path='metadata/groundtruth_strong_label_evaluation_set.csv',
)
"""Dataset instance for the testing dataset."""


def to_dataset(name):
    """Return the Dataset instance corresponding to the given name.

    Args:
        name (str): Name of dataset.

    Returns:
        The Dataset instance corresponding to the given name.
    """
    if name == 'training':
        return training_set
    elif name == 'validation':
        return validation_set
    elif name == 'test':
        return test_set
    return None
