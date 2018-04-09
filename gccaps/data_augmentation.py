import numpy as np

import jams

import muda
from muda import Pipeline
from muda.deformers import Bypass
from muda.deformers import DynamicRangeCompression
from muda.deformers import LinearPitchShift


def transformations(y, sample_rate, n_transforms):
    """Generate transformations for the given audio data.

    Args:
        y (np.ndarray): Input audio data.
        sample_rate (number): Sampling rate of audio.
        n_transforms (tuple): Number of transformations to apply.

    Yields:
        np.ndarray: The transformed audio data.
    """
    # Return empty iterator if number of transforms is zero
    if n_transforms == (0, 0):
        return iter(())

    drc_presets = ['radio', 'film standard']
    n_pitches, n_drc = n_transforms

    # Create deformer for applying transformations
    # It is assumed that n_pitches is non-zero at this point
    deformer = LinearPitchShift(n_samples=n_pitches, lower=-3.5, upper=3.5)
    if n_drc > 0:
        drc = DynamicRangeCompression(preset=drc_presets[:n_drc])
        deformer = Pipeline(steps=[('pitch_shift', Bypass(deformer)),
                                   ('drc', Bypass(drc))])

    # Create JAMS object for input audio and return iterable transforms
    jam = muda.jam_pack(jams.JAMS(), _audio=dict(y=y, sr=sample_rate))
    return map(lambda x: x.sandbox.muda._audio['y'], deformer.transform(jam))


def expand_metadata(metadata):
    """Duplicate the given metadata entries for data augmentation.

    Each metadata entry, which corresponds to a dataset example, is
    copied for every transformation that should be applied to the
    example. This is so that the new metadata structure reflects the
    augmented dataset. The copies are placed next to the original.

    Args:
        metadata (tuple): The metadata structure to expand.

    Returns:
        tuple: The expanded metadata structure.

    See Also:
        :func:`utils.read_metadata`
    """
    names, target_values = metadata
    new_names, new_target_values = [], []

    for i, count in enumerate(transform_counts(target_values)):
        # Calculate number of copies (including original)
        n_copies = (count[0] + 1) * (count[1] + 1)
        for _ in range(n_copies):
            new_names.append(names[i])
            new_target_values.append(target_values[i])

    return new_names, np.array(new_target_values)


def transform_counts(target_values):
    """Return a generator for the transformation counts of a dataset.

    Args:
        target_values (list): A list of target values for a dataset
            indicating which class each example belongs to.

    Yields:
        tuple: A tuple of the form ``(n_pitches, n_drc)``.
    """
    n_examples = np.sum(target_values, axis=0).astype(int)
    for y in target_values:
        # Determine how many transformations should be applied to this
        # example based on the smallest class it belongs to.
        min_n_examples = min(n_examples[label]
                             for label, value in enumerate(y) if value)

        yield transform_count(min_n_examples)


def transform_count(n_examples):
    """Return the number of transformations that should be applied to
    each example in a class.

    This function returns the number of pitch and dynamic range
    compression (DRC) transformations that should be applied to a class
    in which the total number of examples is equal to `n_examples`. The
    idea is that small classes should have a larger number of
    transformations applied in order to balance the dataset.

    Args:
        n_examples (int): The number of examples in the class.

    Returns:
        tuple: A tuple of the form ``(n_pitches, n_drc)``.
    """
    if n_examples < 500:
        return (8, 3)
    elif n_examples < 999:
        return (5, 2)
    elif n_examples < 4999:
        return (2, 1)
    elif n_examples < 9999:
        return (2, 0)

    return (0, 0)
