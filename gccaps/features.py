import os.path
import datetime as dt

import scipy.signal as signal

import h5py
import librosa
import numpy as np
from tqdm import tqdm

import data_augmentation as aug
import utils


def extract_dataset(dataset_path,
                    file_names,
                    extractor,
                    clip_duration,
                    output_path,
                    recompute=False,
                    n_transforms_iter=None,
                    ):
    """Extract features from the audio clips in a dataset.

    Args:
        dataset_path (str): Path of directory containing dataset.
        file_names (list): List of file names for the audio clips.
        extractor: Class instance for feature extraction.
        clip_duration: Duration of a reference clip in seconds. Used to
            ensure all feature vectors are of the same length.
        output_path: File path of output HDF5 file.
        recompute (bool): Whether to extract features that already exist
            in the HDF5 file.
        n_transforms_iter (iterator): Iterator for the number of
            transformations to apply for each example. If data
            augmentation should be disabled, set this to ``None``.
            Otherwise, ensure that `file_names` has been expanded as if
            by calling :func:`data_augmentation.expand_metadata`.
    """
    # Create/load the HDF5 file to store the feature vectors
    with h5py.File(output_path, 'a') as f:
        size = len(file_names)  # Size of dataset

        # Create/load feature vector dataset and timestamp dataset
        feats_shape = (size,) + extractor.output_shape(clip_duration)
        feats = f.require_dataset('F', feats_shape, dtype=np.float32)
        timestamps = f.require_dataset('timestamps', (size,),
                                       dtype=h5py.special_dtype(vlen=bytes))

        transforms = iter(())

        for i, name in enumerate(tqdm(file_names)):
            # Skip if existing feature vector should not be recomputed
            if timestamps[i] and not recompute:
                next(transforms, None)
                continue

            # Generate next transform or, if iterator is empty, load
            # the next audio clip from disk. Note that the iterator will
            # always be empty if data augmentation (DA) is disabled.
            if next(transforms, None) is None:
                # Load audio file from disk
                path = os.path.join(dataset_path, name)
                x, sample_rate = librosa.load(path, sr=None)

                # Create new transform generator if DA is enabled
                if n_transforms_iter:
                    transforms = aug.transformations(
                        x, sample_rate, next(n_transforms_iter))

            # Compute feature vector using extractor
            vec = extractor.extract(x, sample_rate)
            vec = utils.pad_truncate(vec, feats_shape[1])

            # Save to dataset
            feats[i] = vec
            # Record timestamp in ISO format
            timestamps[i] = dt.datetime.now().isoformat()


def load_features(path):
    """Load feature vectors from the specified HDF5 file.

    Args:
        path (str): Path to the HDF5 file.

    Returns:
        np.ndarray: Array of feature vectors.
    """
    with h5py.File(path, 'r') as f:
        return np.array(f['F'])


class LogmelExtractor(object):
    """Feature extractor for logmel representations.

    A logmel feature vector is a spectrogram representation that has
    been scaled using a Mel filterbank and a log nonlinearity.

    Args:
        sample_rate (number): Target resampling rate.
        n_window (int): Number of bins in each spectrogram frame.
        n_overlap (int): Amount of overlap between frames.
        n_mels (int): Number of Mel bands.

    Attributes:
        sample_rate (number): Target resampling rate.
        n_window (int): Number of bins in each spectrogram frame.
        n_overlap (int): Amount of overlap between frames.
        mel_fb (np.ndarray): Mel fitlerbank matrix.
    """

    def __init__(self,
                 sample_rate=16000,
                 n_window=1024,
                 n_overlap=512,
                 n_mels=64,
                 ):
        self.sample_rate = sample_rate
        self.n_window = n_window
        self.n_overlap = n_overlap

        # Create Mel filterbank matrix
        self.mel_fb = librosa.filters.mel(sr=sample_rate,
                                          n_fft=n_window,
                                          n_mels=n_mels,
                                          )

    def output_shape(self, clip_duration):
        """Determine the shape of a logmel feature vector.

        Args:
            clip_duration (number): Duration of the input time-series
                signal given in seconds.

        Returns:
            tuple: The shape of a logmel feature vector.
        """
        n_samples = clip_duration * self.sample_rate
        hop = self.n_window - self.n_overlap
        n_frames = (n_samples - self.n_window) // hop + 1
        return (n_frames, self.mel_fb.shape[0])

    def extract(self, x, sample_rate):
        """Transform the given signal into a logmel feature vector.

        Args:
            x (np.ndarray): Input time-series signal.
            sample_rate (number): Sampling rate of signal.

        Returns:
            np.ndarray: The logmel feature vector.
        """
        # Resample to target sampling rate
        x = librosa.resample(x, sample_rate, self.sample_rate)

        # Compute spectrogram using Hamming window
        _, _, sxx = signal.spectrogram(x=x,
                                       window='hamming',
                                       nperseg=self.n_window,
                                       noverlap=self.n_overlap,
                                       detrend=False,
                                       mode='magnitude',
                                       )
        # Transform to Mel frequency scale
        sxx = np.dot(sxx.T, self.mel_fb.T)
        # Apply log nonlinearity
        return np.log(sxx + 1e-8).astype(np.float32)
