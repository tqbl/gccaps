import os

from sklearn import metrics

from keras.callbacks import Callback
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

import capsnet
import config as cfg
import data_generator
import evaluation
import inference


def train(tr_x, tr_y, val_x, val_y):
    """Train a neural network using the given training set.

    Args:
        tr_x (np.ndarray): Array of training examples.
        tr_y (np.ndarray): Target values of the training examples.
        val_x (np.ndarray): Array of validation examples.
        val_y (np.ndarray): Target values of the validation examples.
    """
    # Create model and print summary
    model = capsnet.gccaps(input_shape=tr_x.shape[1:],
                           n_classes=tr_y.shape[1])
    model.summary()

    # Use Adam SGD optimizer
    optimizer = Adam(lr=cfg.learning_rate['initial'])
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  )

    # Create the appropriate callbacks to use during training
    callbacks = _create_callbacks()

    # Set a large value for `n_epochs` if early stopping is used
    n_epochs = cfg.n_epochs
    if n_epochs < 0:
        n_epochs = 10000

    # Train model using class-balancing generator
    batch_size = cfg.batch_size
    generator = data_generator.balanced_generator(tr_x, tr_y, batch_size)
    steps_per_epoch = len(tr_x) // batch_size
    return model.fit_generator(generator=generator,
                               steps_per_epoch=steps_per_epoch,
                               epochs=n_epochs,
                               callbacks=callbacks,
                               validation_data=(val_x, val_y),
                               use_multiprocessing=False,
                               )


class EERLogger(Callback):
    """A callback for computing the equal error rate (EER).

    At the end of each epoch, the EER is computed and logged for the
    predictions of the validation dataset.
    """
    def on_epoch_end(self, epoch, logs=None):
        """Compute the EER of the validation set predictions."""
        x, y_true = self.validation_data[:2]
        y_pred = self.model.predict(x)
        rate = evaluation.compute_eer(y_true.flatten(), y_pred.flatten())

        # Log the computed value
        logs = logs or {}
        logs['val_eer'] = rate


class MAPLogger(Callback):
    """A callback for computing the mean average precision at k (MAP@k).

    At the end of each epoch, the MAP is computed and logged for the
    predictions of the validation dataset. It is assumed that the ground
    truths are single-label.

    Args:
        k (int): The maximum number of predicted elements.

    Attributes:
        k (int): The maximum number of predicted elements.
    """
    def __init__(self, k=3):
        super(MAPLogger, self).__init__()

        self.k = k

    def on_epoch_end(self, epoch, logs=None):
        """Compute the MAP of the validation set predictions."""
        x, y_true = self.validation_data[:2]
        y_pred = self.model.predict(x)
        map_k = evaluation.compute_map(y_true, y_pred, self.k)

        # Log the computed value
        logs = logs or {}
        logs['val_map'] = map_k


class F1ScoreLogger(Callback):
    """A callback for computing the F1 score.

    At the end of each epoch, the F1 score is computed and logged for
    the predictions of the validation dataset.

    Args:
        threshold (float): Threshold used to binarize predictions.

    Attributes:
        threshold (float): Threshold used to binarize predictions.
    """
    def __init__(self, threshold=0.5):
        super(F1ScoreLogger, self).__init__()

        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        """Compute the F1 score of the validation set predictions."""
        x, y_true = self.validation_data[:2]
        y_pred = self.model.predict(x)
        y_pred_b = inference.binarize_predictions_2d(y_pred, self.threshold)
        f1_score = metrics.f1_score(y_true, y_pred_b, average='micro')

        # Log the computed value
        logs = logs or {}
        logs['val_f1_score'] = f1_score


def _create_callbacks():
    """Create a list of training callbacks.

    Up to four callbacks are included in the list:
      * A callback for saving models.
      * A callback for using TensorBoard.
      * An optional callback for learning rate decay.
      * An optional callback for early stopping.

    Returns:
        list: List of Keras callbacks.
    """
    # Create callbacks for computing various metrics and logging them
    callbacks = [F1ScoreLogger(), EERLogger(), CSVLogger(cfg.history_path)]

    # Create callback to save model after every epoch
    model_path = cfg.model_path
    path = os.path.join(model_path, 'gccaps.{epoch:02d}-{val_acc:.4f}.hdf5')
    callbacks.append(ModelCheckpoint(filepath=path,
                                     monitor='val_acc',
                                     verbose=0,
                                     save_best_only=False,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1,
                                     ))

    # Create callback for TensorBoard logs
    callbacks.append(TensorBoard(cfg.log_path, batch_size=cfg.batch_size))

    lr_decay = cfg.learning_rate['decay']
    if lr_decay < 1.:
        # Create callback to decay learning rate
        def _lr_schedule(epoch, lr):
            decay = epoch % cfg.learning_rate['decay_rate'] == 0
            return lr * lr_decay if decay else lr
        callbacks.append(LearningRateScheduler(schedule=_lr_schedule))

    if cfg.n_epochs == -1:
        # Create callback to use an early stopping condition
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       ))

    return callbacks
