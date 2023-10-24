
# Description: Helper functions for the Speech Enhancement with Neural Post Processing project

# imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import glob


# autotune for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

# decode tfrecords
def decode_tf_records(seralized_example):
    feature_description = {
        "voicefixer": tf.io.FixedLenFeature([], tf.string),
        "produced": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(seralized_example, feature_description)

    voicefixer, _ = tf.audio.decode_wav(example["voicefixer"], desired_channels=-1)
    produced, _ = tf.audio.decode_wav(example["produced"], desired_channels=-1)
    
    return voicefixer, produced

# shift samples to align signals
def shift_samples_x(x, y):
    """Shifts the samples of x (voicefixer) by 441 samples to the right to match the target y. (Voicefixer delay)"""
    x = tf.concat([tf.zeros([441,1]), x[:-441]], 0)
    return x, y

# load and preprocess dataset
def load_and_preprocess_dataset(path, config, train_dataset=bool):
    """Load and preprocess dataset from tfrecords"""

    paths = glob.glob(path)
    dataset = tf.data.TFRecordDataset(paths)
    dataset = dataset.map(decode_tf_records, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(shift_samples_x, num_parallel_calls=AUTOTUNE)
    
    # set number of elements in dataset from config
    if config['DS'] == None:
        dataset = dataset
    else:
        dataset = dataset.take(config['DS'])

    if train_dataset:
        # count elements in train_dataset
        print('Dataset:')
        print(f'Number of elements in train datasets: {len([d for d in dataset])}')

        # data augmentation
        # sample shift
        # waveform auf 0 setzen
        # volume change
        # pitch shift
        # time stretch
        #train_dataset = train_dataset.map(lambda x, y:  ,num_parallel_calls=AUTOTUNE)

    dataset = dataset.shuffle(config['shuffle_buffer_size']).batch(config['batch_size']).prefetch(buffer_size=AUTOTUNE)
    return dataset


def make_logdir():
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # make directory if not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# get audiofile from dataset and use as input for prediction
def set_1_speechfile(dataset, log_dir, config):

    dataset = dataset.unbatch().as_numpy_iterator()
    speech_for_predicition = []
    target = []
    for i, sample in enumerate(dataset):
        speech_for_predicition.append(sample[0])
        target.append(sample[1])
        break
    # # normalize speechfile
    speech_for_predicition = speech_for_predicition[0]
    target = target[0]
    
    # plot voicefixer
    plt.figure(figsize=(8, 4))
    x = np.arange(0, len(speech_for_predicition)/int(config['sr']), 1/int(config['sr']))
    plt.plot(x, speech_for_predicition)
    plt.title('Speechfile used for prediction (Voicefixer)')
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.savefig(log_dir + '/_audiofile_for_prediction.png')
    plt.close()

    # plot target
    plt.figure(figsize=(8, 4))
    x = np.arange(0, len(target)/int(config['sr']), 1/int(config['sr']))
    plt.plot(x, target)
    plt.title('target speechfile (Produced))')
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.savefig(log_dir + '/_target.png')
    plt.close()

    # save voicefixer
    speech_tf = tf.audio.encode_wav(tf.cast(speech_for_predicition, tf.float32), int(config['sr']))
    tf.io.write_file(log_dir + '/_audiofile_for_prediction' + '.wav', speech_tf)
    # save target
    target_tf = tf.audio.encode_wav(tf.cast(target, tf.float32), int(config['sr']))
    tf.io.write_file(log_dir + '/_target' + '.wav', target_tf)

    return speech_for_predicition

# audio conversion
def float2pcm(sig, dtype='int16'):
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

# Custom Loss Functions
class TFSpectralConvergence(tf.keras.layers.Layer):
    """Spectral convergence loss."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y_mag, x_mag):
        """Calculate forward propagation.
        Args:
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return tf.norm(y_mag - x_mag, ord="fro", axis=(-2, -1)) / tf.norm(y_mag, ord="fro", axis=(-2, -1))
    

class TFLogSTFTMagnitude(tf.keras.layers.Layer):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y_mag, x_mag):
        """Calculate forward propagation.
        Args:
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral magnitude loss value.
        """
        return tf.abs(tf.math.log(y_mag) - tf.math.log(x_mag))
    

class TFSTFT(tf.keras.layers.Layer):
    """STFT loss module."""

    def __init__(self, frame_length=600, frame_step=120, fft_length=1024):
        """Initialize."""
        super().__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.spectral_convergenge_loss = TFSpectralConvergence()
        self.log_stft_magnitude_loss = TFLogSTFTMagnitude()

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value (pre-reduce).
            Tensor: Log STFT magnitude loss value (pre-reduce).
        """

        # squeeze last dim if exists
        x = tf.squeeze(x, axis=-1)
        y = tf.squeeze(y, axis=-1)

        # # transpose to (B, T)
        # x = tf.transpose(x, perm=(1, 0))
        # y = tf.transpose(y, perm=(1, 0))

        x_mag = tf.abs(tf.signal.stft(signals=x,
                                      frame_length=self.frame_length,
                                      frame_step=self.frame_step,
                                      fft_length=self.fft_length))
        y_mag = tf.abs(tf.signal.stft(signals=y,
                                      frame_length=self.frame_length,
                                      frame_step=self.frame_step,
                                      fft_length=self.fft_length))

        # add small number to prevent nan value.
        # compatible with pytorch version.
        # x_mag = tf.math.sqrt(x_mag ** 2 + 1e-7)
        # y_mag = tf.math.sqrt(y_mag ** 2 + 1e-7)
        x_mag = tf.clip_by_value(tf.math.sqrt(x_mag ** 2 + 1e-7), 1e-7, 1e3)
        y_mag = tf.clip_by_value(tf.math.sqrt(y_mag ** 2 + 1e-7), 1e-7, 1e3)

        sc_loss = self.spectral_convergenge_loss(y_mag, x_mag)
        mag_loss = self.log_stft_magnitude_loss(y_mag, x_mag)

        return sc_loss, mag_loss
    

class TFMultiResolutionSTFT(tf.keras.layers.Layer):
    """Multi resolution STFT loss module."""

    # Original
    def __init__(self,
                 fft_lengths=[1024, 2048, 512],
                 frame_lengths=[600, 1200, 240],
                 frame_steps=[120, 240, 50],):

    # # Modified
    # def __init__(self,
    #      fft_lengths=[1024, 2048, 4096],
    #      frame_lengths=[1024, 2048, 4096],
    #      frame_steps=[256, 512, 1024],):

        """Initialize Multi resolution STFT loss module.
        Args:
            frame_lengths (list): List of window lengths. 
            frame_steps (list): List of hop sizes.
            fft_lengths (list): List of FFT sizes.
        """
        super().__init__()
        assert len(frame_lengths) == len(frame_steps) == len(fft_lengths)
        self.stft_losses = []
        for frame_length, frame_step, fft_length in zip(frame_lengths, frame_steps, fft_lengths):
            self.stft_losses.append(TFSTFT(frame_length, frame_step, fft_length))

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(y, x)
            sc_loss += tf.reduce_mean(sc_l)
            mag_loss += tf.reduce_mean(mag_l)

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return mag_loss
        #return sc_loss, mag_loss
    
class CustomLoss(tf.keras.losses.Loss):
    """custom loss calculated from MAE/MSE and Multiresolution STFT loss"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.stft = TFMultiResolutionSTFT()


    def call(self, y_true, y_pred):
        # define the loss functions
        mae = self.mae(y_true, y_pred) # l1 loss
        stft = self.stft(y_true, y_pred) # multi resolution stft loss

        if self.config['loss_func'] == 'stft':
            return (stft)
        elif self.config['loss_func'] == 'mix':
            return (stft + (mae/4))



