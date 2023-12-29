
# Description: Helper functions for the Speech Enhancement with Neural Post Processing project

# imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import glob
from scipy.signal import butter, filtfilt
import librosa


# autotune for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE


# --------------------------------------------
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



# --------------------------------------------
# augment funcs

def volume_change(x, y):
    """Changes the volume of the input x by a random factor between 0.3 and 1.0."""
    factor = tf.random.uniform([], 0.3, 1.0)
    x = x * factor
    return x, y

def samples_to_zero(x, y):
    """Sets some samples of the input x to zero."""
    # get random number for number of samples to set to zero
    num_samples = tf.random.uniform([], 200, 800, dtype=tf.int32)
    # get random number for start index
    start_index = tf.random.uniform([], 0, 100000, dtype=tf.int32)
    # set samples to zero
    x = tf.concat([x[:start_index], tf.zeros([num_samples, 1]), x[start_index+num_samples:]], 0)
    return x, y

def roll(x, y):
    """Rolls the input x by a random number of samples between 0 and 400."""
    # get random number for number of samples to roll
    num_samples = tf.random.uniform([], 0, 400, dtype=tf.int32)
    # roll
    x = tf.roll(x, num_samples, axis=0)
    y = tf.roll(y, num_samples, axis=0)
    return x, y

def bandpass_filter(x, y):
    """Applies a bandpass filter to the input x."""

    # convert tensor to numpy array
    x = x.numpy()
    x = np.transpose(x)
 
    sr = 44100
    order = 6
    # get random number for lowcut and highcut
    lowcut = 180
    highcut = 7500

    # getting filter coefficients and filter audio
    b, a = butter(order, [lowcut, highcut], fs=sr, btype='band')
    filtered = filtfilt(b, a, x)
    filtered = np.transpose(filtered)
    # convert numpy array to tensor
    x = tf.convert_to_tensor(filtered, dtype=tf.float32)
    #x = tf.cast(filtered, tf.float32)

    return x, y

def bandpass_wrapper(x, y):
    x, y = tf.py_function(func=bandpass_filter, inp=[x, y], Tout=[tf.float32, tf.float32])
    # x.set_shape([132300, 1])
    # y.set_shape([132300, 1])
    return x, y




#--------------------------------------------
# augment voicefixer-dataset

def resample(x, y):
    """Resamples the dataset to samplerate in config."""

    target_sr = 22050

    # remove last dimension
    x, y = np.squeeze(x, axis=-1), np.squeeze(y, axis=-1)

    # resample
    x = librosa.resample(x, orig_sr=44100, target_sr=target_sr)
    y = librosa.resample(y, orig_sr=44100, target_sr=target_sr)

    # add last dimension
    x, y = np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)

    return x, y

# resample with librosa
def resample_wrapper(x, y):
    """Resamples the dataset to samplerate in config."""
    x, y = tf.numpy_function(func=resample, inp=[x, y], Tout=[tf.float32, tf.float32])
    return x, y


# shift samples to align signals
def shift_samples_22050(x, y):
    """Shifts the samples of x (voicefixer) by 60 samples to the right to match the target y. (Voicefixer delay)
    60 samples with 22050Hz samplerate are 2.72ms.
    75 samples with 44100Hz samplerate are 1.7ms."""

    samples = 60
    #samples = 75
    x = tf.concat([tf.zeros([samples,1]), x[:-samples]], 0)
    return x, y

    # shift samples to align signals
def shift_samples_44100(x, y):
    """Shifts the samples of x (voicefixer) by 60 samples to the right to match the target y. (Voicefixer delay)
    60 samples with 22050Hz samplerate are 2.72ms.
    75 samples with 44100Hz samplerate are 1.7ms."""

    samples = 75
    x = tf.concat([tf.zeros([samples,1]), x[:-samples]], 0)
    return x, y


# take only 1s of audio
def take_1s_22050(x, y):
    """Takes only the first second of the audio."""
    # get random number for start index
    length = tf.cast(tf.shape(x)[0], tf.int32)
    sr = tf.constant(22050, tf.int32)  #sample per second
    #sr = tf.cast(tf.divide(length, tf.constant(3)), tf.int32)  #sample per second
    # get random number for start index
    start_index = tf.random.uniform([], 0, length-sr, dtype=tf.int32)
    # take only 1s of audio
    x = x[start_index:start_index+sr]
    y = y[start_index:start_index+sr]
    return x, y

def take_1s_44100(x, y):
    """Takes only the first second of the audio."""
    # get random number for start index
    length = tf.cast(tf.shape(x)[0], tf.int32)
    sr = tf.constant(44100, tf.int32)  #sample per second
    #sr = tf.cast(tf.divide(length, tf.constant(3)), tf.int32)  #sample per second
    # get random number for start index
    start_index = tf.random.uniform([], 0, length-sr, dtype=tf.int32)
    # take only 1s of audio
    x = x[start_index:start_index+sr]
    y = y[start_index:start_index+sr]
    return x, y


#--------------------------------------------


# load and preprocess dataset
def load_and_preprocess_dataset(path, config, dset) -> tf.data.Dataset:
    """Load and preprocess dataset from tfrecords
       
     Args:
        path (str): Path to tfrecords.
        config (dict): Config.
        augmentation (bool): Augment dataset.
        
    Returns:
        tf.data.Dataset: Dataset."""
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    paths = glob.glob(path)
    dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(decode_tf_records, num_parallel_calls=AUTOTUNE)

    # set number of elements in different datasets from config
    if config['DS'] == None and dset == 'train':
        dataset = dataset
    elif config['DS'] == None and dset == 'test':
        dataset = dataset
    elif config['DS'] == None and dset == 'valid':
        dataset = dataset.take(10)

    elif config['DS'] != None and dset == 'train':
        dataset = dataset.take(config['DS'])
    elif config['DS'] != None and dset == 'test':
        dataset = dataset.take(np.ceil(config['DS']/10))
    elif config['DS'] != None and dset == 'valid':
        dataset = dataset.take(10)


    # resample if samplerate is not 22050
    if config['sr'] == 44100:
        # take only 1s of audio
        dataset = dataset.map(take_1s_44100, num_parallel_calls=AUTOTUNE, deterministic=False)
        # shift samples to align signals if voicefixer dataset
        dataset = dataset.map(shift_samples_44100, num_parallel_calls=AUTOTUNE, deterministic=False)
    elif config['sr'] == 22050:
        # convert samplerate to 22050
        dataset = dataset.map(resample_wrapper, num_parallel_calls=AUTOTUNE, deterministic=False)
        # take only 1s of audio
        dataset = dataset.map(take_1s_22050, num_parallel_calls=AUTOTUNE, deterministic=False)
        # shift samples to align signals if voicefixer dataset
        dataset = dataset.map(shift_samples_22050, num_parallel_calls=AUTOTUNE, deterministic=False)
        

    # shuffle, batch and prefetch
    dataset = dataset.shuffle(config['shuffle_buffer_size']).batch(config['batch_size']).prefetch(buffer_size=1)

    return dataset



# --------------------------------------------
def make_logdir(config):
    log_dir = './logs/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_'+config['model_name']+'_'+config['loss_func']+'_'+str(config['n_epochs'])+'ep_'+str(config['sr'])+'_ds'+str(config['DS'])+'/'
    # make directory if not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# get audiofile from dataset and use as input for prediction
def set_1_speechfile(dataset, log_dir, config):
    """Get audiofile from dataset and use as input for prediction
   
     Args:
        dataset (tf.data.Dataset): Dataset.
        log_dir (str): Path to log directory.
        config (dict): Config.
    
    Returns:
        numpy array: input speechfile.
    """

    dataset = dataset.unbatch().as_numpy_iterator()
    speech_for_predicition = []
    target = []
    for i, sample in enumerate(dataset):
        speech_for_predicition.append(sample[0])
        target.append(sample[1])
        break

    speech_for_predicition = speech_for_predicition[0]
    target = target[0]

    # resample if samplerate is 22050Hz
    if config['sr'] == 22050:
        # resample to 22050
        speech_for_predicition = librosa.resample(speech_for_predicition, orig_sr=44100, target_sr=22050)
        target = librosa.resample(target, orig_sr=44100, target_sr=22050)
    
    # plot input
    plt.figure(figsize=(8, 4))
    x = np.arange(0, len(speech_for_predicition)/int(config['sr']), 1/int(config['sr']))
    plt.plot(x, speech_for_predicition)
    plt.title('Input speechfile')
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.savefig(log_dir + '/_audiofile_for_prediction.png')
    plt.close()

    # plot target
    plt.figure(figsize=(8, 4))
    x = np.arange(0, len(target)/int(config['sr']), 1/int(config['sr']))
    plt.plot(x, target)
    plt.title('target speechfile (Produced)')
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.savefig(log_dir + '/_target.png')
    plt.close()

    # save input
    speech_tf = tf.audio.encode_wav(tf.cast(speech_for_predicition, tf.float32), int(config['sr']))
    tf.io.write_file(log_dir + '/_audiofile_for_prediction' + '.wav', speech_tf)
    # save target
    target_tf = tf.audio.encode_wav(tf.cast(target, tf.float32), int(config['sr']))
    tf.io.write_file(log_dir + '/_target' + '.wav', target_tf)

    return speech_for_predicition



# get audiofile from dataset and use as input for prediction
def pred_plot_save(dataset, model, log_dir, config):
    """Get audiofile from dataset and use as input for prediction
   
     Args:
        dataset (tf.data.Dataset): Dataset.
        model (tf.keras.Model): Model.
        log_dir (str): Path to log directory.
        config (dict): Config.

    """
    # get audios
    dataset = dataset.unbatch().as_numpy_iterator()
    for d in dataset:
        input = d[0]
        target = d[1]
        # get types and shapes ()
        print('input shape: ', input.shape, input.dtype)
        print('target shape: ', target.shape, target.dtype)
        break

    # predict - returns numpy array - (data, channels, 1)
    pred = model.predict(input)
    pred = np.squeeze(pred, axis=-1)        
    print('pred shape: ', pred.shape, pred.dtype)

    
    x = np.arange(0, (pred.shape[0])/int(config['sr']), 1/int(config['sr']))
    # plot waveforms
    plt.plot(x, np.squeeze(input), label="input")
    plt.plot(x, np.squeeze(target), label="target", alpha=0.5)
    plt.plot(x, np.squeeze(pred), label="pred", alpha=0.5)
    plt.legend(loc="upper left")
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    #plt.xlim(0, 0.)
    plt.title(config['model_name'] + ' - ' + config['loss_func'] + ' - ' + str(config['n_epochs']) + ' epochs')
    plt.savefig(log_dir + '/__plot_wav.png')
    plt.close()


    # plot specs
    # calculate stft
    input_stft = librosa.stft(y=np.squeeze(input), n_fft=2048, hop_length=120, win_length=2048)
    target_stft = librosa.stft(y=np.squeeze(target), n_fft=2048, hop_length=120, win_length=2048)
    pred_stft = librosa.stft(y=np.squeeze(pred), n_fft=2048, hop_length=120, win_length=2048)

    # convert to db
    input_db = librosa.amplitude_to_db(np.abs(input_stft), ref=np.max)
    target_db = librosa.amplitude_to_db(np.abs(target_stft), ref=np.max)
    pred_db = librosa.amplitude_to_db(np.abs(pred_stft), ref=np.max)

    # plot three stfts
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,8))
    img1 = librosa.display.specshow(input_db, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set(title='input')

    img2 = librosa.display.specshow(target_db, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].set(title='target')

    img3 = librosa.display.specshow(pred_db, x_axis='time', y_axis='log', ax=ax[2])
    ax[2].set(title='predict')

    # To eliminate redundant axis labels, we'll use "label_outer" on all subplots:
    for ax_i in ax:
        ax_i.label_outer()

    # And we can share colorbars:
    fig.colorbar(img1, ax=[ax[0], ax[1], ax[2]])
    plt.savefig(log_dir + '/__plot_stft.png')
    plt.close()

    # save audio files
    tf.io.write_file(log_dir+'/_input.wav', tf.audio.encode_wav(tf.cast(input, dtype=tf.float32), config['sr']))  
    tf.io.write_file(log_dir+'/_target.wav', tf.audio.encode_wav(tf.cast(target, dtype=tf.float32), config['sr']))
    tf.io.write_file(log_dir+'/_pred.wav', tf.audio.encode_wav(tf.cast(pred, dtype=tf.float32), config['sr']))




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


# --------------------------------------------  
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
        # # check that y_true and y_pred have the same shape
        # if y_mag.shape != x_mag.shape:
        #     raise ValueError(f"y_true and y_pred must have the same shape {y_mag.shape} != {x_mag.shape}")

        # # pad the shorter signal with zeros
        # if tf.shape(y_mag)[1] < tf.shape(x_mag)[1]:
        #     y_mag = tf.pad(y_mag, [[0, 0], [0, tf.abs(tf.shape(y_mag)[1] - tf.shape(x_mag)[1])], [0, 0]])
        # elif tf.shape(y_mag)[1] > tf.shape(x_mag)[1]:
        #     x_mag = tf.pad(x_mag, [[0, 0], [0, tf.abs(tf.shape(x_mag)[1] - tf.shape(y_mag)[1])], [0, 0]])

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

        return sc_loss, mag_loss
        #return sc_loss, mag_loss
    
class CustomLoss(tf.keras.losses.Loss):
    """custom loss calculated from MAE and Multiresolution STFT loss"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.stft = TFMultiResolutionSTFT()


    def call(self, y_true, y_pred):

        # define the loss functions
        mae = self.mae(y_true, y_pred) # l1 loss
        mse = self.mse(y_true, y_pred) # l2 loss
        sc, mag = self.stft(y_true, y_pred) 
        MRstft = sc + mag # multi resolution stft loss

        if self.config['loss_func'] == 'stft':
            return (MRstft)
        elif self.config['loss_func'] == 'mae':
            return mae
        elif self.config['loss_func'] == 'mse':
            return mse
        # mix losses
        elif self.config['loss_func'] == 'mix1':
            return (MRstft + (mae/2))
        # mix2 beste
        elif self.config['loss_func'] == 'mix2':
            return (MRstft + (mae))
        elif self.config['loss_func'] == 'mix3':
            return ((MRstft/2) + (mae))
        elif self.config['loss_func'] == 'mix4':
            return ((MRstft/3) + (mae))



