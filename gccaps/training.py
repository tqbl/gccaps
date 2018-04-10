import os

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

import capsnet
import config as cfg
import data_generator
import evaluation


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


class EqualErrorRate(Callback):
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
    # Create callback for computing the EER after each epoch
    callbacks = [EqualErrorRate()]

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
