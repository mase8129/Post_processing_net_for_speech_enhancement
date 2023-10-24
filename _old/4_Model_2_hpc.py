# Dependencies
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import os.path
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
from scipy.io.wavfile import write
import random
import argparse


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
def set_speechfile(test_dataset, log_dir):


    dataset = test_dataset.unbatch().as_numpy_iterator()
    speech_for_predicition = []
    for i, sample in enumerate(dataset):
        speech_for_predicition.append(sample[0])
        break
    # # normalize speechfile
    speech_for_predicition = speech_for_predicition[0]
    
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
        return (stft)


def build_model(input_shape, config):

    # define model
    model = keras.Sequential(name='PostNet_Conv1D_model2')
    model.add(keras.Input(shape=input_shape, name='input_layer'))

    # add layer 
    model.add(keras.layers.Conv1D(filters=64, kernel_size=16, padding='causal', dilation_rate=4))
    model.add(keras.layers.Activation(config['activation_func']))

    # add layer 
    model.add(keras.layers.Conv1D(filters=32, kernel_size=8, padding='causal', dilation_rate=4))
    model.add(keras.layers.Activation(config['activation_func']))

    # add layer 
    model.add(keras.layers.Conv1D(filters=32, kernel_size=8, padding='causal', dilation_rate=6))
    model.add(keras.layers.Activation(config['activation_func']))

    # add layer 
    model.add(keras.layers.Conv1D(filters=16, kernel_size=8, padding='causal', dilation_rate=8))
    model.add(keras.layers.Activation(config['activation_func']))

    # add layer 
    model.add(keras.layers.Conv1D(filters=16, kernel_size=6, padding='causal'))
    model.add(keras.layers.Activation(config['activation_func']))

    # add layer 
    model.add(keras.layers.Conv1D(filters=8, kernel_size=4, padding='causal'))
    model.add(keras.layers.Activation(config['activation_func']))

    # Add the final Conv1D layer without activation layer
    model.add(keras.layers.Dense(1, name='output_layer'))
    print(f'model output shape: {model.output_shape}')

    return model



#--------------------------------------------

def main():

    # initialize log_dir
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # make directory if not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_Conv1d_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--filter_size', type=int, default=128)
    parser.add_argument('--kernel_size', type=int, default=32)
    parser.add_argument('--activation_func', type=str, default='tanh', choices=['tanh', 'relu', 'sigmoid'])
    parser.add_argument('--DS', type=int, default=None)
    args = parser.parse_args()
    
    # get args to config-dict
    config = vars(args)
    
    # add values to config
    config['shuffle_buffer_size'] = 1000
    config['model_output_channels'] = 1
    config['Datetime'] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(config)
    

    # some values for the model
    input_shape = (3*44100, 1)
    
    # save config to disk
    with open(log_dir + '/config.json', 'w+') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)
    
    
    #--------------------------------------------
    
    # load train tfrecords
    tfrecords_paths = glob.glob('/scratch/marius-s/Dataset/train_tfrecords/*.tfrecords')
    train_dataset = tf.data.TFRecordDataset(tfrecords_paths)

    # if config['DS_take'] is not 'all', then convert to int
    if config['DS'] == None:
        train_dataset = train_dataset
    else:
        train_dataset = train_dataset.take(config['DS'])
    
    train_dataset = train_dataset.map(decode_tf_records, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.map(slicing_audio, num_parallel_calls=AUTOTUNE)
    # count elements in train_dataset
    print(f'Number of elements in train_dataset: {len([d for d in train_dataset])}')
    # batching and shuffling
    train_dataset = train_dataset.shuffle(config['shuffle_buffer_size']).batch(config['batch_size'])
    
    
    # load test tfrecords
    tfrecords_paths = glob.glob('/scratch/marius-s/Dataset/test_tfrecords/*.tfrecords')
    test_dataset = tf.data.TFRecordDataset(tfrecords_paths)

    # if config['DS_take'] is not 'all', then convert to int
    if config['DS'] == None:
        test_dataset = test_dataset
    else:
        test_dataset = test_dataset.take(config['DS'])
    
    test_dataset = test_dataset.map(decode_tf_records, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(slicing_audio, num_parallel_calls=AUTOTUNE)
    # count elements in test_dataset
    print(f'Number of elements in test_dataset: {len([d for d in test_dataset])}')
    # batching and shuffling
    test_dataset = test_dataset.shuffle(config['shuffle_buffer_size']).batch(config['batch_size'])
    
    
    #--------------------------------------------

    # # get speechfile for prediction
    # speech_for_predicition = set_3_speechfiles(test_dataset, log_dir)
    
    # # define custom callback
    # class CustomCallback(keras.callbacks.Callback):
         
    #     # define functions to happen during training after each epoch
    #     def on_epoch_end(self, epoch, logs=None):
            
    #         # save predicted audio file after each epoch to disk
    #         # get audio file from model prediciton
    #         audio = []
    #         for i in range(len(speech_for_predicition)):
    #             speech = self.model.predict(speech_for_predicition[i])
         
    #             # change shape to (len(audio), 1)
    #             speech = tf.squeeze(speech, axis=-1)
    #             speech = tf.squeeze(speech, axis=-1).numpy()
    #             #print(audio.shape)
    #             speech = speech.astype(np.float32)
    #             audio.append(speech)
        
    #         # plot audiofiles
    #         x = np.arange(0, len(audio[0])/44100, 1/44100)
    #         fig, (ax1, ax2, ax3) = plt.subplots(3)
    #         fig.suptitle('predicted speechfiles after epoch ' + str(epoch+1))
    #         ax1.plot(x, audio[0])
    #         ax2.plot(x, audio[1])
    #         ax3.plot(x, audio[2])
    #         plt.xlabel('Time in s')
    #         plt.ylabel('Amplitude')
    #         plt.tight_layout()
    #         plt.savefig(log_dir + '/0_audiofiles_pred_after_epoch'+ str(epoch+1) + '.png')
    #         plt.close()
            
    #         # save audiofiles to disk
    #         for i, file in enumerate(audio):
    #             write(log_dir + '/1_audiofile_pred_after_epoch_' + str(epoch+1) + '_#'+ str(i) + '.wav', int(44100), float2pcm(file))

    #-----------------------------------


    # get model
    model = build_model(input_shape = input_shape, config=config)
    
    # compile model
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate']),
                  loss = CustomLoss(),
                  metrics = tf.keras.losses.MeanSquaredError())
    
    model.summary()    

    # time how long it takes to train the model
    start = datetime.datetime.now()

    # fit model
    history = model.fit(train_dataset,
                        epochs=config['n_epochs'],
                        validation_data=test_dataset)
    
    stop = datetime.datetime.now()
    print(f'Training time: {stop - start}')
    config['Training_time'] = str(stop - start)

    # save model
    model.save('./model.keras')
    
    # save history
    with open(log_dir + '/history.json', 'w+') as fp:
        json.dump(history.history, fp, sort_keys=True, indent=4)
    
    
    #--------------------------------------------
    # plot loss and accuracy
    
    train_loss = history.history['loss']
    eval_loss = history.history['val_loss']
    
    
    # plot loss and accuracy in one figure
    fig1 = plt.figure()
    plt.plot(range(config['n_epochs']), train_loss, label='train')
    plt.plot(range(config['n_epochs']), eval_loss, label='eval')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training with ' 
                                + str(config['n_epochs'])
                                + ' epochs \n batch-size: '
                                + str(config['batch_size']))
    # save plot to disk
    plt.savefig(log_dir + '/loss.png')
    plt.close()
    

# call main    
if __name__ == "__main__":
    main()
    

