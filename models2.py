# imports
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, Conv2D, UpSampling1D, concatenate, Dense, BatchNormalization, GlobalAveragePooling1D, Flatten, Input, Add
from keras.models import Model
import numpy as np


#--------------------------------------------
# define GLU layer
class GLU(tf.keras.layers.Layer):
    def __init__(self, dim=-1, units=0):
        super(GLU, self).__init__()
        self.linear = tf.keras.layers.Dense(units) # Could be used to make Layer trainable
        self.sigmoid = tf.keras.layers.Dense(units, activation="sigmoid")
        self._dim = dim

    def call(self, inputs):
        out, gate = tf.split(inputs, num_or_size_splits=2, axis=self._dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sigmoid': self.sigmoid,
        })
        return config
    

#--------------------------------------------
# define GAN model from HiFi-GAN and SEGAN paper

def build_generator(input_shape, output_shape, config):

    # Generator - Autoencoder model like in SEGAN paper
    Input_Gen = keras.Input(input_shape)

    # Generator - Encoder 
    conv1 = Conv1D(filters=128, kernel_size=32, padding='same', activation='tanh', strides=2)(Input_Gen)
    conv2 = Conv1D(filters=128, kernel_size=32, padding='same', activation='tanh', strides=2)(conv1)
    conv3 = Conv1D(filters=128, kernel_size=32, padding='same', activation='tanh', strides=2)(conv2)
    conv4 = Conv1D(filters=128, kernel_size=32, padding='same', activation='tanh', strides=2)(conv3)

    # Generator - Decoder with skip connections
    # transpose convolutions
    up1 = Conv1D(filters=128, kernel_size=32, padding='same', activation='tanh', strides=2)(UpSampling1D(size=2))(conv4)
    merge1 = concatenate([up1, conv3], axis=-1)
    up2 = Conv1D(filters=128, kernel_size=32, padding='same', activation='tanh', strides=2)(UpSampling1D(size=2))(merge1)
    merge2 = concatenate([up2, conv2], axis=-1)
    up3 = Conv1D(filters=128, kernel_size=32, padding='same', activation='tanh', strides=2)(UpSampling1D(size=2))(merge2)
    merge3 = concatenate([up3, conv1], axis=-1)
    up4 = Conv1D(filters=128, kernel_size=32, padding='same', activation='tanh', strides=2)(UpSampling1D(size=2))(merge3)
    
    # Generator - Output
    model = Dense(output_shape[1], activation='tanh')(up4)

    # output from generator - then mel-spectrogram - then discriminator input
    # mel-spectrogram
    mel = tf.signal.stft(model, frame_length=1024, frame_step=256, fft_length=1024)
    # mel to db
    mel_dB = tf.math.log(tf.abs(mel))
    # mel_db to db_norm

    return model, mel_dB


def build_discriminator(disc_input, disc_input_shape, config):
    # Discriminator
    # leaky relu instead of GLU ?
    Input_Disc = keras.Input(disc_input_shape)(disc_input)

    # Discriminator - Conv1D
    conv1 = Conv2D(filters=32, kernel_size=(3,9), strides=(1,1))(Input_Disc)
    conv1 = BatchNormalization()(conv1)
    conv1 = GLU()(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3,8), strides=(1,2))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = GLU()(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(3,8), strides=(1,2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = GLU()(conv3)

    conv4 = Conv2D(filters=32, kernel_size=(3,6), strides=(1,2))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = GLU()(conv4)

    # Discriminator - Dense
    flat = Flatten()(conv4)
    Output_Disc = Dense(1, activation='sigmoid')(flat)
    #mean pooling
    Output_Disc = Dense(1, activation='sigmoid')(GlobalAveragePooling1D()(conv4))

    return Output_Disc

