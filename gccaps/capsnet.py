import numpy as np

from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.models import Model

from capsulelayers import CapsuleLayer
from capsulelayers import Length
from capsulelayers import PrimaryCap

import gated_conv


def gccaps(input_shape, n_classes):
    """Create a model using the *GCCaps* architecture.

    Args:
        input_shape (tuple): Shape of the input tensor.
        n_classes (int): Number of classes for classification.

    Returns:
        A Keras model of the GCCaps architecture.
    """
    input_tensor = Input(shape=input_shape, name='input_tensor')

    x = Reshape(input_shape + (1,))(input_tensor)

    x = gated_conv.block(x, n_filters=64, pool_size=(2, 2))
    x = gated_conv.block(x, n_filters=64, pool_size=(2, 2))
    x = gated_conv.block(x, n_filters=64, pool_size=(2, 2))

    x = PrimaryCap(x, dim_capsule=4, n_channels=16, kernel_size=3,
                   strides=(1, 2), padding='same')
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.5)(x)

    caps = TimeDistributed(CapsuleLayer(num_capsule=n_classes,
                                        dim_capsule=8, routings=3))(x)
    caps = TimeDistributed(Length(), name='capsule_length')(caps)

    att = Reshape((int(x.shape[1]), -1))(x)
    att = TimeDistributed(Dense(n_classes, activation='sigmoid'),
                          name='attention_layer')(att)

    x = Lambda(_merge, output_shape=(n_classes,),
               name='merge')([caps, att])

    return Model(input_tensor, x, name='GCCaps')


def gccaps_predict(x, model, batch_size=32):
    """Generate output predictions for the given input examples.

    Args:
        x (np.ndarray): Array of input examples.
        model: Keras model of GCCaps architecture.
        batch_size (int): Number of examples in a mini-batch.

    Returns:
        tuple: A tuple containing the audio tagging predictions and
        SED predictions.
    """

    # Compute audio tagging predictions
    at_preds = model.predict(x, batch_size=batch_size)

    # Prepare for sound event detection
    input_tensor = model.get_layer('input_tensor').input
    capsule_output = model.get_layer('capsule_length').output
    func = K.function([input_tensor, K.learning_phase()], [capsule_output])

    # Compute sound event detection predictions
    n_steps = int(np.ceil(len(x) / batch_size))
    sed_preds = [func([x[batch_size*i:batch_size*(i+1)]])[0]
                 for i in range(n_steps)]
    # Transpose so that final dimension is the time axis
    sed_preds = np.transpose(np.concatenate(sed_preds), (0, 2, 1))

    return at_preds, sed_preds


def _merge(inputs):
    """Merge the given pair of inputs across the temporal dimension.

    Args:
        inputs (list): Pair of inputs to merge. Each input should be a
            T x L Keras tensor (excluding batch dimension), where T is
            the temporal dimension and L is the number of classes.

    Returns:
        A Keras tensor (vector) of length L.
    """
    caps, att = inputs
    att = K.clip(att, K.epsilon(), 1.)
    return K.sum(caps * att, axis=1) / K.sum(att, axis=1)
