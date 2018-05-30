import csv

import numpy as np
from sklearn import metrics

import sed_eval

import inference
import utils


def evaluate_audio_tagging(y_true, y_pred, threshold=0.5):
    """Evaluate audio tagging performance.

    Three types of results are returned:

      * Class-wise
      * Macro-averaged
      * Micro-averaged

    The ground truth values and predictions should both be passed in a
    2D array in which the first dimension is the sample axis and the
    second is the class axis.

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of predictions.
        threshold (float): Threshold used to binarize predictions.

    Returns:
        tuple: Tuple containing class scores, macro-averaged scores, and
        micro-averaged scores in that order. Each tuple element is a
        list (see: :func:`compute_scores`).

    Notes:
        The element ordering of `y_true` and `y_pred` must be the same.
    """
    y_pred_b = inference.binarize_predictions_2d(y_pred, threshold)

    # Compute scores for class-wise performance
    class_scores = compute_audio_tagging_scores(y_true, y_pred, y_pred_b)
    # Compute scores for macro-averaged performance
    macro_scores = [np.mean(scores) for scores in class_scores]
    # Compute scores for micro-averaged performance
    micro_scores = compute_audio_tagging_scores(
        y_true, y_pred, y_pred_b, average='micro')

    return class_scores, macro_scores, micro_scores


def compute_audio_tagging_scores(y_true, y_pred, y_pred_b, average=None):
    """Compute prediction scores using several performance metrics.

    The following metrics are used:

      * F1 Score
      * Precision
      * Recall
      * Equal error rate (EER)
      * Receiver operator characteristic (ROC) area under curve (AUC)

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of prediction probabilities.
        y_pred_b (np.ndarray): 2D array of binary predictions.
        average (str): The averaging method. Either ``'macro'``,
            ``'micro'``, or ``None``, where the latter is used to
            disable averaging.

    Returns:
        tuple: List of scores corresponding to the metrics used (in
        the order listed above).

    Notes:
        The element ordering of `y_true`, `y_pred`, and `y_pred_b` must
        be the same.
    """
    # Compute precision and recall scores
    precision = metrics.precision_score(y_true, y_pred_b, average=average)
    recall = metrics.recall_score(y_true, y_pred_b, average=average)
    f1_score = 2 * precision * recall / (precision + recall + 1e-9)

    # Compute equal error rate
    if average is None:
        eer = np.array([compute_eer(y_true[:, i].flatten(),
                                    y_pred[:, i].flatten())
                        for i in range(y_true.shape[1])])
    else:
        eer = compute_eer(y_true.flatten(), y_pred.flatten())

    # Compute area under curve (AUC) score
    auc = metrics.roc_auc_score(y_true, y_pred, average=average)

    return [f1_score, precision, recall, eer, auc]


def write_audio_tagging_results(results, output_path, print_results=True):
    """Write audio tagging results to a CSV file.

    Args:
        results (tuple): Tuple containing class scores, macro-averaged
            scores, and micro-averaged scores in that order.
        output_path (str): File path of the output CSV file.
        print_results (bool): Whether to also print results to console.
    """
    class_scores, macro_scores, micro_scores = results

    with open(output_path, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)

        # Convenience function for writing and printing a row of data
        def _write_row(row, is_header=False, use_separator=False):
            """Write and (optionally) print a row of data."""
            writer.writerow(row)

            if print_results:
                _print_row(row, is_header, use_separator)

        header = ['Class', 'F1 Score', 'Precision', 'Recall', 'EER', 'AUC']
        _write_row(header, is_header=True, use_separator=True)

        # Write class-specific scores
        class_scores = np.array(class_scores).T
        for i, class_i_scores in enumerate(class_scores):
            _write_row([utils.LABELS[i]] + class_i_scores.tolist(),
                       use_separator=(i == len(class_scores) - 1))

        # Write macro-averaged scores
        _write_row(['Macro Average'] + macro_scores)
        # Write micro-averaged scores
        _write_row(['Micro Average'] + micro_scores)


def evaluate_sed(ground_truth, predictions, names, time_resolution=1.0):
    """Evaluate sound event detection performance using sed_eval [1]_.

    The ground truth values and predictions are assumed to be a list of
    *event lists*, where each event list corresponds to a particular
    audio clip. An event list is a list of events, and an event is a
    ``(label, onset, offset)`` tuple.

    Args:
        ground_truth (list): List of ground truth event lists.
        predictions (list): List of predicted event lists.
        names (list): File names of the audio clips.
        time_resolution (float): Resolution of event times.

    Returns:
        An ``sed_eval.SoundEventMetrics`` instance.

    Notes:
        The element ordering of `ground_truth`, `predictions`, and
        `names` must be the same.

    References:
        .. [1] Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen,
               "Metrics for polyphonic sound event detection",
               Applied Sciences, 6(6):162, 2016
    """
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=utils.LABELS, time_resolution=time_resolution)

    # Evaluate the performance for each example
    for i, name in enumerate(names):
        def _event_list(entries):
            """Create an sed_eval-compatible event list."""
            return [{'file': name,
                     'event_label': label,
                     'event_onset': onset,
                     'event_offset': offset}
                    for label, onset, offset in entries]

        gt_event_list = _event_list(ground_truth[i])
        pred_event_list = _event_list(predictions[i])

        segment_based_metrics.evaluate(
            reference_event_list=gt_event_list,
            estimated_event_list=pred_event_list)

    return segment_based_metrics


def compute_eer(y_true, y_pred):
    """Compute the equal error rate (EER).

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of predictions.

    Returns:
        float: The equal error rate.
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)

    # Find the points closest to the true EER point
    points = list(zip(fpr, tpr))
    i = np.argmax(fpr > 1-tpr)
    p1 = points[(i or 1) - 1]
    p2 = points[i]

    # Interpolate between p1 and p2
    if abs(p2[0] - p1[0]) < 1e-6:
        rate = p1[0]
    else:
        gradient = (p2[1] - p1[1]) / (p2[0] - p1[0])
        offset = p1[1] - gradient * p1[0]
        rate = (1 - offset) / (1 + gradient)
    return rate


def compute_map(y_true, y_pred, k=3):
    """Compute the mean average precision at k (MAP@k).

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of predictions.
        k (int): The maximum number of predicted elements.

    Returns:
        float: The mean average precision at k.

    Note:
        This function accepts a 2D array for `y_true`, but it assumes
        the grounds truths are single-label.
    """
    # Compute how the true label ranks in terms of probability
    idx = y_pred.argsort()[:, ::-1].argsort()
    rank = idx[y_true.astype(bool)] + 1

    if len(rank) > len(y_true):
        raise Exception('Multi-label classification not supported')

    return np.sum(1 / rank[rank <= k]) / len(y_true)


def compute_thresholds(y_true, y_pred):
    """Compute the optimal probability thresholds for each class.

    This function computes the precision-recall curve for each class,
    and selects the threshold corresponding to the highest F1 score.

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of predictions.

    Returns:
        list: The optimal per-class probability thresholds.
    """
    thresholds = []
    for i in range(y_true.shape[1]):
        p, r, t = metrics.precision_recall_curve(y_true[:, i],
                                                 y_pred[:, i])
        p = np.array(p)
        r = np.array(r)
        f1_score = 2 * p * r / (p + r + 1e-9)
        thresholds.append(t[np.argmax(f1_score)])

    return thresholds


def _print_row(row, is_header=False, use_separator=False):
    """Print the given row in a tabulated format.

    Args:
        row (list): List of row cells to print.
        is_header (bool): Whether row is a header (non-numeric).
        use_separator (bool): Whether to print a horizontal rule after.
    """
    cell_format = '{:<%d}' if is_header else '{:<%d.3f}'
    row_format = '{:<%d}' + (cell_format * 5)
    row_widths = (32, 11, 11, 11, 11, 5)

    print((row_format % row_widths).format(*row))
    if use_separator:
        print('=' * sum(row_widths))
