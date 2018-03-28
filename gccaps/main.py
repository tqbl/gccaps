import argparse
import glob
import os
import pickle
import sys

import numpy as np

import config as cfg
import utils


def main():
    """Execute a task based on the given command-line arguments.

    This function is the main entry-point of the program. It allows the
    user to extract features, train a model, generate predictions, or
    evaluate predictions using the command-line interface.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    # Add sub-parser for feature extraction
    parser_extract = subparsers.add_parser('extract')
    parser_extract.add_argument('dataset',
                                choices=['training', 'validation', 'test'],
                                )

    # Add sub-parser for training
    subparsers.add_parser('train')

    # Add sub-parser for inference
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('dataset',
                                nargs='?',
                                choices=['validation', 'test'],
                                default='test',
                                )

    # Add sub-parser for evaluation
    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument('task',
                                 nargs='?',
                                 choices=['tagging', 'sed', 'all'],
                                 default='all',
                                 )
    parser_evaluate.add_argument('dataset',
                                 nargs='?',
                                 choices=['validation', 'test'],
                                 default='test',
                                 )

    args = parser.parse_args()

    if args.mode == 'extract':
        extract(cfg.to_dataset(args.dataset))
    elif args.mode == 'train':
        train()
    elif args.mode == 'predict':
        predict(cfg.to_dataset(args.dataset))
    elif args.mode == 'evaluate':
        eval_all = args.task == 'all'
        dataset = cfg.to_dataset(args.dataset)
        if args.task == 'tagging' or eval_all:
            evaluate_audio_tagging(dataset)
        if args.task == 'sed' or eval_all:
            evaluate_sed(dataset)


def extract(dataset):
    """Extract feature vectors from the given dataset.

    Args:
        dataset: Dataset to extract features from.
    """
    import data_augmentation as aug
    import features

    # Use a logmel representation for feature extraction
    extractor = features.LogmelExtractor(cfg.sample_rate, cfg.n_window,
                                         cfg.n_overlap, cfg.n_mels)

    # Prepare for data augmentation if enabled
    file_names, target_values = utils.read_metadata(dataset.metadata_path)
    if dataset == cfg.training_set and cfg.enable_augmentation:
        n_transforms_iter = aug.transform_counts(target_values)
        file_names = aug.expand_metadata((file_names, target_values))[0]
    else:
        n_transforms_iter = None

    # Ensure output directory exists and set file path
    os.makedirs(cfg.extraction_path, exist_ok=True)
    output_path = os.path.join(cfg.extraction_path, dataset.name + '.h5')

    # Generate features for each audio clip in the dataset
    features.extract_dataset(dataset.path,
                             file_names,
                             extractor,
                             cfg.clip_duration,
                             output_path,
                             n_transforms_iter=n_transforms_iter,
                             )


def train():
    """Train the neural network model.

    See Also:
        :func:`training.train`

    Note:
        For reproducibility, the random seed is set to a fixed value.
    """
    import training

    # Ensure output directories exist
    os.makedirs(os.path.dirname(cfg.scaler_path), exist_ok=True)
    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.log_path, exist_ok=True)

    # Load (standardized) input data and target values
    tr_x, tr_y, _ = _load_data(cfg.training_set, is_training=True)
    val_x, val_y, _ = _load_data(cfg.validation_set)

    # Try to create reproducible results
    np.random.seed(cfg.initial_seed)

    training.train(tr_x, tr_y, val_x, val_y)


def predict(dataset):
    """Generate predictions for audio tagging and sound event detection.

    This function uses an ensemble of trained models to generate the
    predictions, with the averaging function being an arithmetic mean.
    Computed predictions are then saved to disk.

    Args:
        dataset: Dataset to generate predictions for.
    """
    import capsnet

    # Load (standardized) input data and associated file names
    test_x, _, names = _load_data(dataset)

    # Predict class probabilities for each model (epoch)
    at_preds, sed_preds = [], []
    for epoch in cfg.prediction_epochs:
        model = _load_model(epoch)
        at_pred, sed_pred = utils.timeit(
            lambda: capsnet.gccaps_predict(test_x, model),
            '[Epoch %d] Predicted class probabilities' % epoch)

        at_preds.append(at_pred)
        sed_preds.append(sed_pred)

    # Average predictions to give an overall output
    total_at_pred = np.mean(at_preds, axis=0)
    total_sed_pred = np.mean(sed_preds, axis=0)

    # Ensure output directory exists and set file path format
    os.makedirs(os.path.dirname(cfg.predictions_path), exist_ok=True)
    predictions_path = cfg.predictions_path.format('%s', dataset.name)

    # Write predictions to disk
    utils.write_predictions(names, total_at_pred, predictions_path % 'at')
    utils.write_predictions(names, total_sed_pred, predictions_path % 'sed')


def evaluate_audio_tagging(dataset):
    """Evaluate the audio tagging predictions and write results.

    Args:
        dataset: Dataset for retrieving ground truth.
    """
    import evaluation

    _, y_true = utils.read_metadata(dataset.metadata_path)
    path = cfg.predictions_path.format('at', dataset.name)
    _, y_pred = utils.read_predictions(path)

    scores = evaluation.evaluate_audio_tagging(
        y_true, y_pred, threshold=cfg.at_threshold)

    # Ensure output directory exist and set file path
    os.makedirs(os.path.dirname(cfg.results_path), exist_ok=True)
    output_path = cfg.results_path.format('at', dataset.name)

    # Evaluate tagging performance and write results
    evaluation.write_audio_tagging_results(scores, output_path)


def evaluate_sed(dataset):
    """Evaluate the sound event detection predictions and print results.

    Args:
        dataset: Dataset for retrieving ground truth.
    """
    import evaluation
    import inference

    names, ground_truth = utils.read_metadata(dataset.metadata_path,
                                              weakly_labeled=False)

    # Load predictions and convert to event list format
    path = cfg.predictions_path.format('sed', dataset.name)
    _, y_pred = utils.read_predictions(path)
    resolution = cfg.clip_duration / y_pred.shape[1]
    y_pred_b = inference.binarize_predictions_3d(y_pred,
                                                 threshold=cfg.sed_threshold,
                                                 n_dilation=cfg.sed_dilation,
                                                 n_erosion=cfg.sed_erosion)
    predictions = inference.generate_event_lists(y_pred_b, resolution)

    # Ensure output directory exist and set file path
    os.makedirs(os.path.dirname(cfg.results_path), exist_ok=True)
    output_path = cfg.results_path.format('sed', dataset.name)

    # Evaluate SED performance and write results
    metrics = evaluation.evaluate_sed(ground_truth, predictions, names)
    with open(output_path, 'w') as f:
        f.write(metrics.result_report_overall())
        f.write(metrics.result_report_class_wise())


def _load_data(dataset, is_training=False):
    """Load input data, target values and file names for a dataset.

    The input data is assumed to be a dataset of feature vectors. These
    feature vectors are standardized using a scaler that is either
    loaded from disk (if it exists) or computed on-the-fly. The latter
    is only possible if the input data is training data, which is
    indicated by the `is_training` parameter.

    Target values and file names are read from the metadata file.

    Args:
        dataset: Structure encapsulating dataset information.
        training (bool): Whether the input data is training data.

    Returns:
        x (np.ndarray): The input data.
        y (np.ndarray): The target values.
        names (list): The associated file names.
    """
    import data_augmentation as aug
    import features

    features_path = os.path.join(cfg.extraction_path, dataset.name + '.h5')
    x = utils.timeit(lambda: features.load_features(features_path),
                     'Loaded features of %s dataset' % dataset.name)

    # Load scaler from file if cached, or else compute it.
    scaler_path = cfg.scaler_path
    if os.path.exists(scaler_path) or not is_training:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = utils.timeit(lambda: utils.compute_scaler(x),
                              'Computed standard scaler')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    x = utils.timeit(lambda: utils.standardize(x, scaler),
                     'Standardized %s features' % dataset.name)

    names, y = utils.timeit(lambda: utils.read_metadata(dataset.metadata_path),
                            'Loaded %s metadata' % dataset.name)
    if dataset == cfg.training_set and cfg.enable_augmentation:
        names, y = aug.expand_metadata((names, y))

    return x, y, names


def _load_model(epoch=-1):
    """Load model based on specified epoch number.

    Args:
        epoch (int): Epoch number of the model to load or -1 for the
            last saved model.

    Returns:
        An instance of a Keras model.
    """
    import keras.models

    from capsulelayers import CapsuleLayer
    from capsulelayers import Length
    from gated_conv import GatedConv

    model_path = cfg.model_path
    if epoch == -1:
        model_path = glob.glob(os.path.join(model_path, '*.hdf5'))[-1]
    else:
        model_path = glob.glob(os.path.join(
            model_path, '*.%.02d*.hdf5' % epoch))[0]

    custom_objects = {
        'GatedConv': GatedConv,
        'CapsuleLayer': CapsuleLayer,
        'Length': Length,
    }
    return keras.models.load_model(model_path, custom_objects)


if __name__ == '__main__':
    sys.exit(main())
