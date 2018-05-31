GCCaps - Capsule Routing for Sound Event Detection
==================================================

This is the source code for the system described in the paper 'Capsule Routing
for Sound Event Detection'.

Requirements
------------

This software requires Python 3.5 or higher. To install the dependencies, run::

    pip install -r requirements.txt

The main functionality of this software also requires the DCASE 2017 Task 4
datasets, which may be downloaded here_. After acquiring the datasets, modify
``gccaps/config/paths.py`` accordingly.

For example::

    _root_dataset_path = ('/path/to/datasets')
    """str: Path to root directory containing input audio clips."""

    training_set = Dataset(
        name='training_weak',
        path=os.path.join(_root_dataset_path, 'audio/training'),
        metadata_path=os.path.join(_root_dataset_path, 'metadata/training.csv'),
    )
    """Dataset instance for the weak training dataset."""

You may also want to change the work path::

    work_path = '/path/to/workspace'
    """str: Path to parent directory containing program output."""

.. _here: http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/download#task4---large-scale-weakly-supervised-sound-event-detection-for-smart-cars

Usage
-----

In this section, the various commands are described. Using this software, the
user is able to extract feature vectors, train the network, generate
predictions using the trained network, and evaluate the predictions.

Feature Extraction
^^^^^^^^^^^^^^^^^^

To extract feature vectors, run::

    python gccaps/main.py extract <training/validation/test>

See ``gccaps/config/logmel.py`` for tweaking the parameters.

Training
^^^^^^^^

To train the GCCaps network, run::

    python gccaps/main.py train

See ``gccaps/config/training.py`` for tweaking the parameters, or
``gccaps/training.py`` for further modifications.

Prediction
^^^^^^^^^^

To generate predictions, run::

    python gccaps/main.py predict [validation/test]

See ``gccaps/config/predictions.py`` to modify which models (corresponding to
different epochs) are selected for generating the predictions. By default, the
top five models based on their F-score on the validation set are chosen.

Evaluation
^^^^^^^^^^

To evaluate the predictions, run::

    python gccaps/main.py evaluate <tagging/sed/all> [validation/test] [--thresholds]

See ``gccaps/config/predictions.py`` for tweaking the parameters.

The ``--thresholds`` flag indicates that 'optimal' class-wise probability
thresholds should be computed and recorded. That is, precision-recall curves
will be computed for each class, and the threshold corresponding to the highest
F-score will be saved to disk. The idea is to compute the thresholds for the
validation set and use them for the test set. To use the 'optimal' thresholds,
set ``at_threshold`` or ``sed_threshold`` in ``gccaps/config/predictions.py``
to ``-1``. This is included for experimental purposes.
