import numpy as np

import utils


def binarize_predictions_2d(predictions, threshold=0.5):
    """Convert prediction probabilities to binary values.

    This function is intended for audio tagging predictions. The
    predictions should be passed in a 2D array in which the first
    dimension is the sample axis and the second is the class axis. If no
    classes exceed the threshold for a sample, the class with the
    highest probability is set to 1.

    Args:
        predictions (np.ndarray): 2D array of predictions.
        threshold (float): Threshold used to determine binary value.

    Returns:
        np.ndarray: Binarized prediction values.

    Examples:
        >>> binarize_predictions(np.array([[0.7, 0.3], [0.2, 0.4]))
        array([1, 0], [0, 1])
    """
    binary_preds = (predictions > threshold).astype(int)

    # Choose the class with the highest prediction probability if no
    # classes were predicted for a particular sample.
    for i, _ in enumerate(predictions):
        if not binary_preds[i].any():
            binary_preds[i, np.argmax(predictions[i, :])] = 1

    return binary_preds


def binarize_predictions_3d(predictions, threshold=0.5,
                            n_dilation=1, n_erosion=1):
    """Convert prediction probabilities to binary values.

    This function is intended for sound event detection predictions. The
    predictions should be passed in a 3D array in which the first
    dimension is the sample axis, the second is the class axis, and the
    third is the time axis. To reduce fragmentation and reduce noise, an
    operation similar to morphological closing is applied.

    Args:
        predictions (np.ndarray): 2D array of predictions.
        threshold (float): Threshold used to determine binary value.
        n_dilation (int): A sequence of zeros of this length or less
            will be 'filled'.
        n_erosion (int): A sequence of ones of this length or less in
            isolation will be removed.

    Returns:
        np.ndarray: Binarized prediction values.
    """
    binary_preds = (predictions > threshold).astype(int)

    return np.array([[_erode(_dilate(label_pred, n_dilation), n_erosion)
                      for label_pred in sample_pred]
                     for sample_pred in binary_preds])


def generate_event_lists(predictions, resolution):
    """Generate a list of event lists from the given predictions.

    An *event list* is a list of detected sound events for a particular
    audio clip, where an event is represented as a tuple of the form
    ``(label, onset, offset)``. This function expects a 3D array of
    binarized predictions in which the first dimension is the sample
    axis. Thus, it returns an event list for each audio clip.

    Args:
        predictions (np.ndarray): 3D array of binarized predictions.
        resolution (float): Number of seconds per time slice.

    Returns:
        list: A list of event lists.

    See Also:
        :func:`binarize_predictions_3d`
    """
    def _generate_events(pred, label):
        # Pad array so that non-zero edges are detected
        pred = np.concatenate([[0], pred, [0]])
        # Find locations of boundaries (0->1 and 1->0 transitions)
        locs = np.where(np.diff(pred))[0]
        # Compute onset and offset times in seconds
        onsets = locs[::2] * resolution
        offsets = (locs[1::2] - 1) * resolution
        return zip([utils.LABELS[label]] * len(onsets), onsets, offsets)

    return [[event for label, label_pred in enumerate(sample_pred)
             for event in _generate_events(label_pred, label)]
            for sample_pred in predictions]


def _dilate(x, n_window=1):
    """Apply an operation similar to morphological dilation.

    A sequence of zeros of length less than or equal to `n_window` is
    considered a 'hole' and gets 'filled' by this function. This
    excludes zeros that are at the edge of the input vector.

    Args:
        x (array-like): Binary vector to appy operation to.
        n_window (int): Size of the hole-filling window.
    """
    y = np.copy(x)

    # Find locations of 1 and calculate their cumulative difference,
    # which enables us to find the holes. For example, if `diffs` is
    # [1,1,1,4,1,1,2,1], there are holes of size 3 and 1.
    locs = np.where(y)[0]
    diffs = np.diff(locs)
    for i, diff in enumerate(diffs):
        if diff > 1 and diff <= n_window + 1:
            y[(locs[i] + 1):(locs[i] + diff)] = 1

    return y


def _erode(x, n_window=1):
    """Apply an operation similar to morphological erosion.

    A sequence of ones of length less than or equal to `n_window` is
    considered noise and gets removed by this function.

    Args:
        x (array-like): Binary vector to appy operation to.
        n_window (int): Size of the noise-removing window.
    """
    y = np.copy(x)

    # Find locations of 1s and calculate their cumulative difference,
    # which enables us to find the noise. The array of locations is
    # padded so that 1s at the edge of the input vector can be detected.
    locs = np.concatenate([np.where(y)[0], [0]])
    diffs = np.diff(locs)

    count = 0
    for i, diff in enumerate(diffs):
        if diff == 1:
            count += 1
        else:
            if count < n_window:
                y[(locs[i] - count):(locs[i] + 1)] = 0
            count = 0

    return y
