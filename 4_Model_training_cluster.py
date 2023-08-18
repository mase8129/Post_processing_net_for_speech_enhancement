# Dependencies

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models


# Helper libraries
import os
import os.path
import glob
import librosa
import librosa.display
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as pd
import pprint
import datetime        
from scipy.io.wavfile import write


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


def slicing_audio(voicefixer, produced):
    # generate random integer between 0 and 10*44100-3*44100
    random_int = tf.random.uniform(shape=[], minval=0, maxval=7*44100, dtype=tf.int32)
    # slice audio
    voicefixer = voicefixer[random_int:random_int+3*44100]
    produced = produced[random_int:random_int+3*44100]
    return voicefixer, produced
 

#--------------------------------------------
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


#--------------------------------------------
# get audiofile from dataset and use as input for prediction
def set_speechfile(test_dataset, log_dir):


    dataset = test_dataset.unbatch().as_numpy_iterator()
    speech_for_predicition = []
    for i, sample in enumerate(dataset):
        speech_for_predicition.append(sample[0])
        break
    # # normalize speechfile
    speech_for_predicition = speech_for_predicition[0]
    speech_for_predicition = librosa.util.normalize(speech_for_predicition)
    
    # save speech file used for prediction
    # save plot to disk
    plt.figure(figsize=(8, 4))
    x = np.arange(0, len(speech_for_predicition)/44100, 1/44100)
    plt.plot(x, speech_for_predicition)
    plt.title('Speechfile used for prediction')
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.savefig(log_dir + '/_audiofile_for_prediction.png')
    plt.close()
    
    # save audiofile to disk
    write(log_dir + '/_audiofile_for_prediction' + '.wav', int(44100), float2pcm(speech_for_predicition))
   
    return speech_for_predicition


#--------------------------------------------
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


#--------------------------------------------
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
            Tensor: Spectral convergence loss value.
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

        x = tf.transpose(x, perm=[0, 2, 1])
        y = tf.transpose(y, perm=[0, 2, 1])        

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


class CustomLoss(tf.keras.losses.Loss):
    """custom loss calculated from MAE and Multiresolution STFT loss"""

    def __init__(self):
        super().__init__()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.stft = TFMultiResolutionSTFT()


    def call(self, y_true, y_pred):
        mae = self.mae(y_true, y_pred)
        stft = self.stft(y_true, y_pred)
        return ((mae+stft) / 2)


#--------------------------------------------

# some values for training
config = {
    'batch_size': 32,
    'shuffle_buffer_size': 1000,
    'n_epochs': 2,
    'learning_rate': 0.003,
    # Filter:Kernel = 4:1 (see hifi-gan paper)
    # 'filter_size': 64,
    # 'kernel_size': 5,
    'filter_size': 10,
    'kernel_size': 10,
    'dropout_rate': 0.5
}

# some values for the model
input_shape = (3*44100, 1)
output_channels = 1


# save config to disk
with open('./config_cluster.json', 'w+') as fp:
    json.dump(config, fp, sort_keys=True, indent=4)


#--------------------------------------------


# load train tfrecords
tfrecords_paths = glob.glob(os.path.join(os.pardir, 'Dataset/train_tfrecords/*.tfrecords'))
train_dataset = tf.data.TFRecordDataset(tfrecords_paths[:1])
train_dataset = train_dataset.map(decode_tf_records, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(slicing_audio, num_parallel_calls=AUTOTUNE)
# count elements in train_dataset
print(f'Number of elements in train_dataset: {len([d for d in train_dataset])}')
# batching and shuffling
train_dataset = train_dataset.shuffle(config['shuffle_buffer_size']).batch(config['batch_size'])



# load test tfrecords
tfrecords_paths = glob.glob(os.path.join(os.pardir, 'Dataset/test_tfrecords/*.tfrecords'))
test_dataset = tf.data.TFRecordDataset(tfrecords_paths[:1])
test_dataset = test_dataset.map(decode_tf_records, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(slicing_audio, num_parallel_calls=AUTOTUNE)
# count elements in test_dataset
print(f'Number of elements in test_dataset: {len([d for d in test_dataset])}')
# batching and shuffling
test_dataset = test_dataset.shuffle(config['shuffle_buffer_size']).batch(config['batch_size'])


#--------------------------------------------

# build model with 12 layers
def build_model(input_shape):

    # define model
    model = keras.Sequential(name='PostNet_Conv1D')
    model.add(keras.Input(shape=input_shape))

    # add layer 
    model.add(keras.layers.Conv1D(filters=config['filter_size'], kernel_size=config['kernel_size'], padding='same'))
    # add dropout layer, batch normalization and activation layer
    #model.add(keras.layers.Dropout(config['dropout_rate']))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('tanh'))

    # Add the remaining Conv1D layers
    for _ in range(11):
        model.add(keras.layers.Conv1D(filters=config['filter_size'], kernel_size=config['kernel_size'], padding='same'))
        # add dropout layer, batch normalization and activation layer
        #model.add(keras.layers.Dropout(config['dropout_rate']))
        #model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('tanh'))

    # Add the final Conv1D layer without activation layer
    model.add(keras.layers.Conv1D(output_channels, kernel_size=config['kernel_size'], padding='same'))
    #model.add(keras.layers.Dropout(config['dropout_rate']))
    #model.add(keras.layers.BatchNormalization())
    print(f'model output shape: {model.output_shape}')

    return model


#--------------------------------------------


# initialize log_dir
log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# make directory if not exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


#--------------------------------------------
# define callbacks

save_callback = keras.callbacks.ModelCheckpoint(
    filepath='./model_checkpoint',
    save_best_only=False,
    monitor='val_loss',
    verbose=0)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir= log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    write_steps_per_second=True,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=1)

# early_stopping_callback = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     verbose=1)


# get speechfile for prediction
speech_for_predicition = set_speechfile(test_dataset, log_dir)

# # define custom callback
class CustomCallback(keras.callbacks.Callback):
     
    # define functions to happen during training after each epoch
    def on_epoch_end(self, epoch, logs=None):
        
        # save predicted audio file after each epoch to disk
        # get audio file from model prediciton
        audio = self.model.predict(speech_for_predicition)

        # change shape to (len(audio), 1)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.squeeze(audio, axis=-1).numpy()
        #print(audio.shape)

        # normalize audio with numpy
        #audio = librosa.util.normalize(audio).astype(np.float32)
        audio = audio.astype(np.float32)

        # save plot to disk
        plt.figure(figsize=(8, 4))
        x = np.arange(0, len(audio)/44100, 1/44100)
        plt.plot(x, audio)
        plt.title('Audiofile')
        plt.xlabel('Time in s')
        plt.ylabel('Amplitude')
        plt.savefig(log_dir + '/_audiofile_epoch' + str(epoch+1) + '.png')
        plt.close()

        # write audio file to disk (16-bit PCM WAV)
        write(log_dir + '/_audiofile_epoch' + str(epoch+1) + '.wav', 44100, float2pcm(audio))


#--------------------------------------------
# get model
model = build_model(input_shape = input_shape)

# compile model
model.compile(optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate']),
              loss = CustomLoss(),
              metrics = tf.keras.losses.MeanAbsoluteError())

model.summary()

#--------------------------------------------


# fit model
history = model.fit(train_dataset,
                    epochs=config['n_epochs'],
                    validation_data=test_dataset,
                    callbacks=[save_callback, tensorboard_callback, CustomCallback()])

# save model
model.save('./model.keras')

# save history
with open('./history.json', 'w+') as fp:
    json.dump(history.history, fp, sort_keys=True, indent=4)


# call def to save audio to summaries
save_audio_to_summaries(log_dir)