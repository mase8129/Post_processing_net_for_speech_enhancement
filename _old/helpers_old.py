import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from scipy.io.wavfile import write
import glob
import matplotlib.pyplot as plt
import librosa
import librosa.display

# funcs
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


def shift_samples_x(x, y):
    """Shifts the samples of x (voicefixer) by 441 samples to the right to match the target y. (Voicefixer delay)"""
    x = tf.concat([tf.zeros([441,1]), x[:-441]], 0)
    return x, y


def calc_STFT(voicefixer, produced):

    # calc STFT
    voicefixer = tf.signal.stft(signals=voicefixer,
                                frame_length=2048,
                                frame_step=256,
                                fft_length=2048)
    
    produced = tf.signal.stft(signals=produced,
                              frame_length=2048,
                              frame_step=256,
                              fft_length=2048)
    
    # # squeeze dims
    # voicefixer = tf.squeeze(voicefixer, axis=0)
    # produced = tf.squeeze(produced, axis=0)

    
    voicefixer_dB = tf.math.log(tf.abs(voicefixer) + 1e-7)
    produced_dB = tf.math.log(tf.abs(produced) + 1e-7)
    
    return voicefixer_dB, produced_dB


# helper func
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


# get audiofile from dataset and use as input for prediction
def set_1_specfile(test_dataset, log_dir):

    dataset = test_dataset.unbatch().as_numpy_iterator()
    for_predicition = []
    target = []
    for i, sample in enumerate(dataset):
        for_predicition.append(sample[0])
        target.append(sample[1])
        break
    
    spec_for_predicition = for_predicition[0]
    spec_target = target[0]

    # get shape
    print(f'shape of spec_for_predicition: {spec_for_predicition.shape}')
    print(f'shape of spec_target: {spec_target.shape}')

    # plot waveshow of speechfile used for prediction
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(spec_for_predicition, sr=44100, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('spec_for_predicition')
    plt.xlabel('Time in s')
    plt.ylabel('Frequency in Hz')
    plt.savefig(log_dir + '/_spec_for_prediction.png')
    plt.close()

    # plot target
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(spec_target, sr=44100, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('spec_target')
    plt.xlabel('Time in s')
    plt.ylabel('Frequency in Hz')
    plt.savefig(log_dir + '/_spec_target.png')
    plt.close()

    return spec_for_predicition


# get audiofile from dataset and use as input for prediction
def set_1_speechfile(test_dataset, log_dir, config):

    dataset = test_dataset.unbatch().as_numpy_iterator()
    speech_for_predicition = []
    target = []
    for i, sample in enumerate(dataset):
        speech_for_predicition.append(sample[0])
        target.append(sample[1])
        break
    # # normalize speechfile
    speech_for_predicition = speech_for_predicition[0]
    target = target[0]
    
    # save speech file used for prediction
    # save plot to disk
    plt.figure(figsize=(8, 4))
    x = np.arange(0, len(speech_for_predicition)/int(config['sr']), 1/int(config['sr']))
    plt.plot(x, speech_for_predicition)
    plt.title('Speechfile used for prediction')
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.savefig(log_dir + '/_audiofile_for_prediction.png')
    plt.close()

    # plot target
    plt.figure(figsize=(8, 4))
    x = np.arange(0, len(target)/int(config['sr']), 1/int(config['sr']))
    plt.plot(x, target)
    plt.title('target speechfile')
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.savefig(log_dir + '/_target.png')
    plt.close()

    # save speechfile
    speech_tf = tf.audio.encode_wav(tf.cast(speech_for_predicition, tf.float32), int(config['sr']))
    tf.io.write_file(log_dir + '/_audiofile_for_prediction' + '.wav', speech_tf)
    # save target
    target_tf = tf.audio.encode_wav(tf.cast(target, tf.float32), int(config['sr']))
    tf.io.write_file(log_dir + '/_target' + '.wav', target_tf)

    return speech_for_predicition



# get audiofile from dataset and use as input for prediction
def set_3_speechfiles(dataset, log_dir):

    # unbatch dataset and save 3 files to list
    DS = dataset.unbatch().as_numpy_iterator()
    # get length of dataset
    length = len([d for d in dataset.unbatch()])    
    # generate 3 random numbers
    random.seed(42)
    random_numbers = random.sample(range(0, length), 3)
    # initate list for speechfiles
    all_speechfiles = []
    
    for i, sample in enumerate(DS):
        all_speechfiles.append(sample[0])

    speech_for_predicition = all_speechfiles[random_numbers[0]], all_speechfiles[random_numbers[1]], all_speechfiles[random_numbers[2]]

    # plot speechfiles used for prediction
    x = np.arange(0, len(speech_for_predicition[0])/44100, 1/44100)
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('3 Speechfiles used for prediction')
    ax1.plot(x, speech_for_predicition[0])
    ax2.plot(x, speech_for_predicition[1])
    ax3.plot(x, speech_for_predicition[2])
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(log_dir + '/0_audiofiles_for_prediction.png')
    plt.close()
    
    # save audiofiles to disk
    for i, file in enumerate(speech_for_predicition):
        write(log_dir + '/1_audiofile_for_prediction_' + str(i) + '.wav', int(44100), float2pcm(file))
        
   
    return speech_for_predicition

# save audio files from logs to tf.summary.audio and event files for tensorboard
def save_audio_to_summaries(log_dir):
    
        # get audio files from logs
        fps = glob.glob(log_dir + '/*.wav')
        path_audiosummary = log_dir + '/audiosummary/'

        # for each audio file
        for idx, fp in enumerate(fps):
            
            # write audiosummary of one audio file to disk
             writer = tf.summary.create_file_writer(path_audiosummary)
             with writer.as_default():
                
                # load audio file as tensor
                file = tf.io.read_file(fp)
                audio = tf.audio.decode_wav(file, desired_channels=1)

                # write audio file to tf.summary.audio
                name = fp.split('/')[-1]
                tf.summary.audio(name , tf.expand_dims(audio[0], 0), int(44100), step=idx)
                writer.flush()




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

        # x = tf.transpose(x, perm=[0, 2, 1])
        # y = tf.transpose(y, perm=[0, 2, 1])
        x = tf.squeeze(x, axis=-1)
        y = tf.squeeze(y, axis=-1)

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
        x_mag = tf.math.sqrt(x_mag ** 2 + 1e-7)
        y_mag = tf.math.sqrt(y_mag ** 2 + 1e-7)

        sc_loss = self.spectral_convergenge_loss(y_mag, x_mag)
        mag_loss = self.log_stft_magnitude_loss(y_mag, x_mag)

        return sc_loss, mag_loss
    
class TFMultiResolutionSTFT(tf.keras.layers.Layer):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_lengths=[1024, 2048, 512],
                 frame_lengths=[600, 1200, 240],
                 frame_steps=[120, 240, 50],):
        """Initialize Multi resolution STFT loss module.
        Args:
            frame_lengths (list): List of FFT sizes.
            frame_steps (list): List of hop sizes.
            fft_lengths (list): List of window lengths.
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

    def __init__(self):
        super().__init__()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.stft = TFMultiResolutionSTFT()


    def call(self, y_true, y_pred):
        mae = self.mae(y_true, y_pred)
        mse = self.mse(y_true, y_pred)
        stft = self.stft(y_true, y_pred)
        return (stft)
