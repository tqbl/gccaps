prediction_epochs = list(range(10, 15))
"""list: Models to select for prediction based on epoch number."""

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
