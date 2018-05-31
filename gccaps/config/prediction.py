prediction_epochs = 'val_f1_score'
"""Specification for which models (epochs) to select for prediction.

Either a list of epoch numbers or a string specifying how to select the
epochs. The valid string values are ``'val_acc'`` and ``'val_eer'``.
"""

at_threshold = 0.35
"""number: Number for thresholding audio tagging predictions.

A value of -1 indicates that thresholds should be loaded from disk.

See Also:
    :func:`evaluation.compute_thresholds`
"""

sed_threshold = 0.6
"""number: Number for thresholding sound event detection predictions.

A value of -1 indicates that thresholds should be loaded from disk.

See Also:
    :func:`evaluation.compute_thresholds`
"""

sed_dilation = 10
"""int: Dilation parameter for binarizing predictions.

See Also:
    :func:`inference.binarize_predictions_3d`
"""

sed_erosion = 5
"""int: Erosion parameter for binarizing predictions.

See Also:
    :func:`inference.binarize_predictions_3d`
"""
