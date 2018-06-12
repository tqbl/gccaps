import keras.backend as K
import keras.initializers as initializers
from keras.layers import Conv2D
from keras.layers import Layer
from keras.layers import Lambda
from keras.layers import Reshape


class CapsuleLayer(Layer):
    """A Keras layer implementing capsule routing [1]_.

    Args:
        n_capsules (int): Number of output capsules.
        dim_capsule (int): Number of units per output capsule.
        routings (int): Number of routing iterations.
        use_bias (bool): Whether to use a bias vector.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias weights.
        kwargs: Other layer keyword arguments.

    Attributes:
        n_capsules (int): Number of output capsules.
        dim_capsule (int): Number of units per output capsule.
        routings (int): Number of routing iterations.
        use_bias (bool): Whether to use a bias vector.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias weights.

    References:
        .. [1] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing
               between capsules,” in Adv. Neural Inf. Process. Syst.
               (NIPS), Long Beach, CA, 2017, pp. 3859–3869.
    """
    def __init__(self, n_capsules, dim_capsule, routings=3, use_bias=False,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)

        self.n_capsules = n_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        """Create the layer weights."""
        self.n_input_capsules = input_shape[1]
        self.dim_input_capsule = input_shape[2]

        self.W = self.add_weight(shape=(self.n_capsules,
                                        self.n_input_capsules,
                                        self.dim_capsule,
                                        self.dim_input_capsule),
                                 initializer=self.kernel_initializer,
                                 name='W')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.n_capsules,),
                                        initializer=self.bias_initializer,
                                        name='bias')

        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        """Apply transformation followed by capsule routing."""
        # Create dimension for output capsules and tile along this dim
        # (None, *n_capsules*, n_input_capsules, dim_input_capsules)
        inputs_tiled = K.tile(K.expand_dims(inputs, 1),
                              [1, self.n_capsules, 1, 1])

        # Apply linear transformation to compute prediction vectors
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]),
                              elems=inputs_tiled)
        # Add bias to prediction vectors if specified
        if self.use_bias:
            inputs_hat = K.bias_add(inputs_hat, self.bias,
                                    data_format='channels_first')

        # Initialize logit variables to zero
        b = K.zeros(shape=[K.shape(inputs_hat)[0],
                           self.n_capsules,
                           self.n_input_capsules])

        # Apply routing algorithm
        for i in range(self.routings):
            # Compute coupling coefficients
            c = K.softmax(b, axis=1)
            # Apple squashing function
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))
            # Update logits by computing agreement
            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output."""
        return (input_shape[0], self.n_capsules, self.dim_capsule)

    def get_config(self):
        """Return the config of the layer."""
        config = super(CapsuleLayer, self).get_config()
        config['n_capsules'] = self.n_capsules
        config['dim_capsule'] = self.dim_capsule
        config['routings'] = self.routings
        config['kernel_initializer'] = self.kernel_initializer
        config['use_bias'] = self.use_bias
        return config


def primary_capsules(x, n_channels, dim_capsule, kernel_size=(3, 3), **kwargs):
    """Apply a convolution followed by a squashing function.

    Args:
        x (tensor): Input tensor to transform.
        n_channels (int): Number of channels per capsule.
        dim_capsule (int): Number of activation units per capsule.
        kernel_size (int or tuple): Size of convolution kernel.
        kwargs: Other layer keyword arguments.

    Returns:
        A Keras tensor with shape ``(None, -1, dim_capsule)``.
    """
    x = Conv2D(n_channels * dim_capsule, kernel_size, **kwargs)(x)
    x = Reshape((-1, dim_capsule))(x)
    return Lambda(squash)(x)


def squash(x, axis=-1):
    """Apply a squashing nonlinearity as described in [1]_.

    Args:
        x (tensor): Input tensor to transform.
        axis (int): Axis along which squashing is applied.

    Returns:
        A Keras tensor of the resulting output.
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) \
        / K.sqrt(s_squared_norm + K.epsilon())
    return scale * x


def length(x):
    """Compute the Euclidean lengths of capsules.

    Args:
        x (tensor): Tensor of capsules.

    Returns:
        A Keras tensor of lengths.
    """
    return K.sqrt(K.sum(K.square(x), -1))
