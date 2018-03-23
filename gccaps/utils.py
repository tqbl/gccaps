from collections import OrderedDict
import csv
import os.path
import pickle
import time

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

import numpy as np


LABELS = ['Train horn',
          'Air horn, truck horn',
          'Car alarm',
          'Reversing beeps',
          'Bicycle',
          'Skateboard',
          'Ambulance (siren)',
          'Fire engine, fire truck (siren)',
          'Civil defense siren',
          'Police car (siren)',
          'Screaming',
          'Car',
          'Car passing by',
          'Bus',
          'Truck',
          'Motorcycle',
          'Train',
          ]
"""Ordered list of class labels."""

LABELS_DICT = {s: i for i, s in enumerate(LABELS)}
"""Dictionary to map labels to their integer values."""


def read_metadata(path, weakly_labeled=True):
    """Read from the specified metadata file.

    The metadata file is assumed to be a CSV file describing the audio
    clips of a particular dataset. The supported format is::

        file_name<tab>onset<tab>offset<tab>label

    Note that each row specifies a single label, so multi-label audio
    clips contain multiple entries.

    If `weakly_labeled` is set to ``True``, the target values that this
    function returns are binary vectors, giving a binary matrix over all
    the audio clips. Otherwise, the target values are event lists, where
    an event is a ``(label, onset, offset)`` tuple.

    Args:
        path (str): Path to metadata file.
        weakly_labeled (bool): Whether the data is weakly-labeled.

    Returns:
        tuple: 2-tuple containing:

        * **names** (*list*): File names of the audio clips.
        * **target_values** (*np.ndarray*): Ground truth. Either a
          binary matrix or a list of event lists.
    """
    y = OrderedDict()
    with open(path, 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            # Extract metadata from row
            name = 'Y' + row[0]
            onset = float(row[1])
            offset = float(row[2])
            label = row[3]

            if name not in y:
                y[name] = []

            # Add event to list, where an 'event' is simply the class
            # label for weakly-labeled data and a tuple otherwise.
            if weakly_labeled:
                y[name].append(LABELS_DICT[label])
            else:
                y[name].append((label, onset, offset))

    names = list(y.keys())
    target_values = list(y.values())

    # Convert target values to binary matrix
    if weakly_labeled:
        target_values = MultiLabelBinarizer().fit_transform(target_values)

    return names, target_values


def pad_truncate(x, length):
    """Pad or truncate an array to a specified length.

    Args:
        x (array): Input array.
        length (int): Target length.

    Returns:
        array: The array padded or truncated to the specified length.
    """
    x_len = len(x)
    if x_len > length:
        x = x[:length]
    elif x_len < length:
        padding = np.zeros((length - x_len,) + x.shape[1:])
        x = np.concatenate((x, padding))

    return x


def qualify_path(file_path, qualifier_path):
    """Prepend a 'qualifier' path if the file path is a base name only.

    This function is useful in the scenario that the user provides a
    file name that should be qualified with respect to a directory other
    than the current working directory; namely, the directory specified
    by `qualifier_path`. Prepending the qualifier path to the given
    file name achieves this.

    If `file_path` is not a base name, it is assumed to be qualified
    already, and the function simply returns its value.

    Args:
        file_path (str): File path to prepend to.
        qualifier_path (str): Path that is to be prepended.

    Returns:
        str: Qualified file path.

    Examples:
        >>> qualify_path('foo.p', 'bar')
        'bar/foo.p'

        >>> qualify_path('baz/foo.p', 'bar')
        'baz/foo.p'
    """
    if file_path != os.path.basename(file_path):
        return file_path

    return os.path.join(qualifier_path, file_path)


def compute_scaler(x):
    r"""Compute mean and standard deviation values for the given data.

    The array `x` is assumed to be a 3D array in which only the last
    dimension corresponds to the components of the feature vectors. For
    example, a :math:`100 \times 50 \times 20` array indicates there are
    5000 feature vectors and 20 components per vector. The mean and
    standard deviation values are computed for each component.

    Args:
        x (np.ndarray): 3D array used to compute the parameters.

    Returns:
        StandardScaler: Scaler used for later transformations.
    """
    x = x.reshape((-1, x.shape[-1]))
    return StandardScaler().fit(x)


def standardize(x, scaler):
    r"""Standardize data using the given scaler.

    The array `x` is assumed to be a 3D array in which only the last
    dimension corresponds to the components of the feature vectors. For
    example, a :math:`100 \times 50 \times 20` array indicates there are
    5000 feature vectors and 20 components per vector.

    Each feature vector, :math:`x_i`, is transformed according to

    .. math:: y_i = (x_i - \mu) / \sigma,

    where :math:`\mu` and :math:`\sigma` are mean and standard deviation
    values specified via the `scaler` parameter. Note: Although the
    notation does not suggest it, each variable in the equation is a
    vector and the operations are component-wise.

    Args:
        x (np.ndarray): 3D array to standardize.
        scaler (StandardScaler): Scaler used for transformation.

    Returns:
        np.ndarray: The standardized data.
    """
    shape = x.shape
    x = x.reshape((-1, shape[-1]))
    y = scaler.transform(x)
    return y.reshape(shape)


def read_predictions(path):
    """Read classification predictions from the specified pickle file.

    Args:
        path (str): Path of pickle file.

    Returns:
        tuple: Tuple containing names and predictions.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_predictions(names, preds, output_path, write_csv=True):
    """Write classification predictions to a pickle file and,
    optionally, a CSV file.

    Args:
        names (list): Names of the predicted examples.
        preds (np.ndarray): 2D or 3D array of predictions.
        output_path (str): Output file path.
        write_csv (bool): Whether to also write to a CSV file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump((names, preds), f)

    if write_csv:
        write_predictions_to_csv(names, preds, output_path[:-1] + 'csv')


def write_predictions_to_csv(names, preds, output_path):
    """Write classification predictions to a CSV file.

    Format of each entry is::

        fname<tab>p_1<tab>p_2<tab>...<tab>p_L

    Args:
        names (list): Names of the predicted examples.
        preds (np.ndarray): 2D or 3D array of predictions.
        output_path (str): Output file path.
    """
    if len(preds.shape) < 3:
        preds = np.expand_dims(preds, axis=-1)

    with open(output_path, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for i, name in enumerate(names):
            for pred in preds[i].T:
                writer.writerow([name] + pred.tolist())


def timeit(callback, message):
    """Measure the time taken to execute the given callback.

    This function measures the amount of time it takes to execute the
    specified callback and prints a message afterwards regarding the
    time taken. The `message` parameter provides part of the message,
    e.g. if `message` is 'Executed', the printed message is 'Executed in
    1.234567 seconds'.

    Args:
        callback: Function to execute and time.
        message (str): Message to print after executing the callback.

    Returns:
        The return value of the callback.
    """
    # Record time prior to invoking callback
    onset = time.time()
    # Invoke callback function
    x = callback()

    print('%s in %f seconds' % (message, time.time() - onset))

    return x
