import numpy as np


def balanced_generator(x, y, batch_size=32):
    """Return a generator that creates class-balanced mini-batches.

    The generator yields batches in which there is a 'fair'[1]_ number
    of examples from each class.

    Args:
        x (np.ndarray): Array of training examples to select from.
        y (np.ndarray): Target values of the training examples.
        batch_size (int): Number of examples in a mini-batch.

    Yields:
        tuple: Mini-batch of the form *(batch_x, batch_y)*.

    References:
        .. [1] Y. Xu, Q. Kong, W. Wang, and M. D. Plumbley, "Large-scale
               weakly supervised audio classification using gated
               convolutional neural network," ArXiv e-prints, 2017.
    """
    # Create an index list for each class, e.g. indexes[0] is a list
    # of indexes (locations) for class *0* w.r.t. `y`.
    n_classes = y.shape[1]
    indexes = [np.where(y[:, label] == 1)[0]
               for label in range(n_classes)]

    # Calculate the number of examples per class
    n_examples = np.sum(y, axis=0).astype(int)

    # Compute the probabilities of an example belonging to a particular
    # class being sampled for the mini-batch, e.g. class_p[0] is the
    # probability that an example from class *0* is sampled.
    class_p = [min(n // 1000 + 1, 5) for n in n_examples]
    class_p = np.array(class_p) / sum(class_p)

    offsets = [0] * n_classes
    while True:
        batch_x = np.empty((batch_size,) + x.shape[1:])
        batch_y = np.empty((batch_size, n_classes))

        labels = np.random.choice(n_classes, size=(batch_size,), p=class_p)
        for i, label in enumerate(labels):
            idx = indexes[label][offsets[label]]
            batch_x[i] = x[idx]
            batch_y[i] = y[idx]

            offsets[label] += 1
            if offsets[label] >= n_examples[label]:
                np.random.shuffle(indexes[label])
                offsets[label] = 0

        yield batch_x, batch_y
