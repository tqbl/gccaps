prediction_epochs = 'val_eer'
"""Specification for which models (epochs) to select for prediction.

Either a list of epoch numbers or a string specifying how to select the
epochs. The valid string values are ``'val_acc'`` and ``'val_eer'``.
"""

at_threshold = 0.35
"""float: Number for thresholding audio tagging predictions."""

sed_threshold = 0.6
"""float: Number for thresholding sound event detection predictions."""

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
