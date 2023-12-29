# imports
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, Conv1DTranspose, UpSampling1D, concatenate, Dense, BatchNormalization, GlobalAveragePooling1D, Flatten, Input, Add
from keras.models import Model
import numpy as np
from model_helpers import *
from model_helpers import length_padded


#--------------------------------------------
# # define models

# model to map inputs to outputs
def build_model_small(input_shape, output_shape, config):
    
    model = keras.Sequential(name='small')
    
    # input layer # (Batchsize, 132300, 1)
    model.add(keras.Input(shape=input_shape, name='input_layer'))

    # add output layer
    model.add(keras.layers.Dense(output_shape[1], name='output_layer'))
    print(f'model output shape: {model.output_shape}')

    return model



# 4layer model 
def build_model_mid(input_shape, output_shape, config):

    model = keras.Sequential(name='4layer')

    # input layer # (Batchsize, 132300, 1)
    model.add(keras.Input(shape=input_shape, name='input_layer'))

    # add 1d layer
    model.add(keras.layers.Conv1D(filters=128, kernel_size=32, padding='same'))
    model.add(keras.layers.Activation('tanh'))

    # add 1d layer
    model.add(keras.layers.Conv1D(filters=128, kernel_size=32, padding='same'))
    model.add(keras.layers.Activation('tanh'))

    # add 1d layer
    model.add(keras.layers.Conv1D(filters=128, kernel_size=32, padding='same'))
    model.add(keras.layers.Activation('tanh'))

    # add 1d layer
    model.add(keras.layers.Conv1D(filters=128, kernel_size=32, padding='same'))
    model.add(keras.layers.Activation('tanh'))
    
    # add output layer
    model.add(keras.layers.Dense(output_shape[1], name='output_layer'))
    print(f'model output shape: {model.output_shape}')

    return model


# HiFi-GAN model
def build_model_HiFiGAN(input_shape, output_shape, config):
    """HiFi-GAN model with 12 Conv1D layers"""

    # # define dilation rates with max of 254 from WAVENET
    # dilation_rates = [2**i if i<8 else 254 for i in range(config['n_layers'])]
    # print(f'dilation_rates: {dilation_rates}')

    # define model
    model = keras.Sequential(name='HiFi-GAN')
    # input layer # (Batchsize, 132300, 1)
    model.add(keras.Input(shape=input_shape, name='input_layer'))

    # add layer 
    model.add(keras.layers.Conv1D(filters=128, kernel_size=32, padding='same'))
    model.add(keras.layers.Activation('tanh'))

    # Add Conv1D layers
    for _ in range(12):
        model.add(keras.layers.Conv1D(filters=128, kernel_size=32, padding='same'))
        model.add(keras.layers.Activation('tanh'))

    # Add the final Conv1D layer without activation layer
    model.add(keras.layers.Conv1D(filters=1, kernel_size=1, name='output_layer'))
    print(f'model output shape: {model.output_shape}')

    return model


# HiFi-GAN model
def build_model_HiFiGAN_v1(input_shape, output_shape, config):
    """HiFi-GAN model with skip connections"""

    # define dilation rates with max of 254 from WAVENET
    # dilation_rates = [2**i if i<8 else 254 for i in range(config['n_layers'])]
    # print(f'dilation_rates: {dilation_rates}')

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=128, kernel_size=32, padding='same')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    conv2 = Conv1D(filters=128, kernel_size=32, padding='same')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    conv3 = Conv1D(filters=128, kernel_size=32, padding='same')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    conv4 = Conv1D(filters=128, kernel_size=32, padding='same')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # skip connection
    merge1 = Add()([conv4, conv1])

    conv5 = Conv1D(filters=128, kernel_size=32, padding='same')(merge1)
    conv5 = keras.layers.Activation('tanh')(conv5)
    conv6 = Conv1D(filters=128, kernel_size=32, padding='same')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    conv7 = Conv1D(filters=128, kernel_size=32, padding='same')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    conv8 = Conv1D(filters=128, kernel_size=32, padding='same')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # skip connection
    merge2 = Add()([conv8, conv1])

    conv9 = Conv1D(filters=128, kernel_size=32, padding='same')(merge2)
    conv9 = keras.layers.Activation('tanh')(conv9)
    conv10 = Conv1D(filters=128, kernel_size=32, padding='same')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    conv11 = Conv1D(filters=128, kernel_size=32, padding='same')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    conv12 = Conv1D(filters=128, kernel_size=32, padding='same')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # skip connection
    #merge3 = Add()([conv12, conv1])

    # output layer
    output_layer = keras.layers.Conv1D(filters=1, kernel_size=1, name='output_layer')(conv12)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v1')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model


# HiFi-GAN model
def build_model_HiFiGAN_v2(input_shape, output_shape, config):
    """HiFi-GAN model with skip connections and batch normalization"""

    # define dilation rates with max of 254 from WAVENET
    # dilation_rates = [2**i if i<8 else 254 for i in range(config['n_layers'])]
    # print(f'dilation_rates: {dilation_rates}')

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=128, kernel_size=32, padding='same')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    conv2 = Conv1D(filters=128, kernel_size=32, padding='same')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(filters=128, kernel_size=32, padding='same')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    conv4 = Conv1D(filters=128, kernel_size=32, padding='same')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # skip connection
    merge1 = Add()([conv4, conv1])

    conv5 = Conv1D(filters=128, kernel_size=32, padding='same')(merge1)
    conv5 = keras.layers.Activation('tanh')(conv5)
    conv6 = Conv1D(filters=128, kernel_size=32, padding='same')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv1D(filters=128, kernel_size=32, padding='same')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    conv8 = Conv1D(filters=128, kernel_size=32, padding='same')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # skip connection
    merge2 = Add()([conv8, conv1])

    conv9 = Conv1D(filters=128, kernel_size=32, padding='same')(merge2)
    conv9 = keras.layers.Activation('tanh')(conv9)
    conv10 = Conv1D(filters=128, kernel_size=32, padding='same')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv1D(filters=128, kernel_size=32, padding='same')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    conv12 = Conv1D(filters=128, kernel_size=32, padding='same')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # skip connection
    #merge3 = Add()([conv12, conv1])

    # output layer
    output_layer = keras.layers.Conv1D(filters=1, kernel_size=1, name='output_layer')(conv12)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v2')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model


def build_model_HiFiGAN_v3(input_shape, output_shape, config):
    """HiFi-GAN model with different tryouts"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=128, kernel_size=32, padding='causal')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    conv2 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    conv3 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    conv4 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # skip connection
    #skip1 = Add()([conv4, conv1])

    conv5 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv4)
    conv5 = keras.layers.Activation('tanh')(conv5)
    conv6 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    conv7 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    conv8 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # skip connection
    #skip2 = Add()([conv8, conv1])

    conv9 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv8)
    conv9 = keras.layers.Activation('tanh')(conv9)
    conv10 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    conv11 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    conv12 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # skip connection
    skip3 = Add()([conv12, conv1])
    # batch normalization
    skip3 = BatchNormalization()(skip3)

    # output layer
    output_layer = Dense(1)(skip3)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v3')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model


def build_model_HiFiGAN_v4(input_shape, output_shape, config):
    """HiFi-GAN model with different tryouts"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=128, kernel_size=32, padding='causal')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    conv2 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    conv3 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    conv4 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # batch normalization
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv4)
    conv5 = keras.layers.Activation('tanh')(conv5)
    conv6 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    conv7 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    conv8 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # batch normalization
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv8)
    conv9 = keras.layers.Activation('tanh')(conv9)
    conv10 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    conv11 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    conv12 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # skip connection
    skip3 = Add()([conv12, conv1])
    # batch normalization
    skip3 = BatchNormalization()(skip3)

    # output layer
    output_layer = Dense(1)(skip3)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v4')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model


def build_model_HiFiGAN_v5(input_shape, output_shape, config):
    """HiFi-GAN model with lots of batch normalization"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=128, kernel_size=32, padding='causal')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    # batch normalization
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    # batch normalization
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    # batch normalization
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # batch normalization
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv4)
    conv5 = keras.layers.Activation('tanh')(conv5)
    # batch normalization
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    # batch normalization
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    # batch normalization
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # batch normalization
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv8)
    conv9 = keras.layers.Activation('tanh')(conv9)
    # batch normalization
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    # batch normalization
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    # batch normalization
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # skip connection
    skip3 = Add()([conv12, conv1])
    # batch normalization
    skip3 = BatchNormalization()(skip3)

    # output layer
    output_layer = Dense(1)(skip3)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v5')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model



def build_model_HiFiGAN_v6(input_shape, output_shape, config):
    """model 5 without skip connection"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=128, kernel_size=32, padding='causal')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    # batch normalization
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    # batch normalization
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    # batch normalization
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # batch normalization
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv4)
    conv5 = keras.layers.Activation('tanh')(conv5)
    # batch normalization
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    # batch normalization
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    # batch normalization
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # batch normalization
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv8)
    conv9 = keras.layers.Activation('tanh')(conv9)
    # batch normalization
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    # batch normalization
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    # batch normalization
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # skip connection
    # skip3 = Add()([conv12, conv1])
    # batch normalization
    conv12 = BatchNormalization()(conv12)

    # output layer
    output_layer = Dense(1)(conv12)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v6')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model



def build_model_HiFiGAN_v7(input_shape, output_shape, config):
    """model with bigger kernels"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=32, kernel_size=128, padding='causal')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    # batch normalization
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    # batch normalization
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    # batch normalization
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # batch normalization
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv4)
    conv5 = keras.layers.Activation('tanh')(conv5)
    # batch normalization
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    # batch normalization
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    # batch normalization
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # batch normalization
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv8)
    conv9 = keras.layers.Activation('tanh')(conv9)
    # batch normalization
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    # batch normalization
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    # batch normalization
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # skip connection
    # skip3 = Add()([conv12, conv1])
    # batch normalization
    conv12 = BatchNormalization()(conv12)

    # output layer
    output_layer = Dense(1)(conv12)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v7')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model


def build_model_HiFiGAN_v8(input_shape, output_shape, config):
    """model with bigger kernels and 1 skip connection"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=32, kernel_size=128, padding='causal')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    # batch normalization
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    # batch normalization
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    # batch normalization
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # batch normalization
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv4)
    conv5 = keras.layers.Activation('tanh')(conv5)
    # batch normalization
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    # batch normalization
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    # batch normalization
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # batch normalization
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv8)
    conv9 = keras.layers.Activation('tanh')(conv9)
    # batch normalization
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    # batch normalization
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    # batch normalization
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv1D(filters=32, kernel_size=128, padding='causal')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # skip connection
    skip = Add()([conv12, conv1])
    # batch normalization
    skip = BatchNormalization()(skip)

    # output layer
    output_layer = Dense(1)(skip)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v8')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model


def build_model_HiFiGAN_v82(input_shape, output_shape, config):
    """model with bigger kernels"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=128, kernel_size=16, padding='causal')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    # batch normalization
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=128, kernel_size=16, padding='causal')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    # batch normalization
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(filters=64, kernel_size=32, padding='causal')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    # batch normalization
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(filters=64, kernel_size=32, padding='causal')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # batch normalization
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=64, kernel_size=64, padding='causal')(conv4)
    conv5 = keras.layers.Activation('tanh')(conv5)
    # batch normalization
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(filters=64, kernel_size=64, padding='causal')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    # batch normalization
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv1D(filters=64, kernel_size=128, padding='causal')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    # batch normalization
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv1D(filters=64, kernel_size=128, padding='causal')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # batch normalization
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv1D(filters=32, kernel_size=256, padding='causal')(conv8)
    conv9 = keras.layers.Activation('tanh')(conv9)
    # batch normalization
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv1D(filters=32, kernel_size=256, padding='causal')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    # batch normalization
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv1D(filters=32, kernel_size=256, padding='causal')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    # batch normalization
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv1D(filters=32, kernel_size=256, padding='causal')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # output layer
    output_layer = Dense(1)(conv12)

    # skip connection
    #skip = Add()([output_layer, input_layer])

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v82')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model


def build_model_HiFiGAN_v81(input_shape, output_shape, config):
    """model with bigger kernels and 1 skip connection"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections
    conv1 = Conv1D(filters=128, kernel_size=16, padding='causal')(input_layer)
    conv1 = keras.layers.Activation('tanh')(conv1)
    # batch normalization
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=128, kernel_size=16, padding='causal')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    # batch normalization
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv1D(filters=64, kernel_size=32, padding='causal')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    # batch normalization
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(filters=64, kernel_size=32, padding='causal')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    # batch normalization
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=64, kernel_size=64, padding='causal')(conv4)
    conv5 = keras.layers.Activation('tanh')(conv5)
    # batch normalization
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(filters=64, kernel_size=64, padding='causal')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    # batch normalization
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv1D(filters=64, kernel_size=128, padding='causal')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    # batch normalization
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv1D(filters=64, kernel_size=128, padding='causal')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    # batch normalization
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv1D(filters=32, kernel_size=256, padding='causal')(conv8)
    conv9 = keras.layers.Activation('tanh')(conv9)
    # batch normalization
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv1D(filters=32, kernel_size=256, padding='causal')(conv9)
    conv10 = keras.layers.Activation('tanh')(conv10)
    # batch normalization
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv1D(filters=32, kernel_size=256, padding='causal')(conv10)
    conv11 = keras.layers.Activation('tanh')(conv11)
    # batch normalization
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv1D(filters=32, kernel_size=256, padding='causal')(conv11)
    conv12 = keras.layers.Activation('tanh')(conv12)
    # output layer
    output_layer = Dense(1)(conv12)

    # skip connection
    skip = Add()([output_layer, input_layer])

    # get model
    model = Model(inputs=input_layer, outputs=skip, name='HiFi-GAN_v81')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model



def build_model_HiFiGAN_v812(input_shape, output_shape, config):
    """model with autoencoder structure and more skip connections
    with transposed convolutions and same"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections - encoder -decoder structure  

    # encoder
    conv1 = Conv1D(filters=256, kernel_size=16, padding='same', activation='tanh')(input_layer)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv1D(filters=128, kernel_size=32, padding='same', activation='tanh')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=64, padding='same', activation='tanh')(conv2)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv1D(filters=32, kernel_size=128, padding='same', activation='tanh')(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=16, kernel_size=256, padding='same', activation='tanh')(conv4)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv1D(filters=8, kernel_size=512, padding='same', activation='tanh')(conv5)
    conv6 = BatchNormalization()(conv6)

    # decoder
    conv7 = Conv1DTranspose(filters=8, kernel_size=512, padding='same', activation='tanh')(conv6)
    conv7 = BatchNormalization()(conv7)
    #skip connection
    conv7 = Add()([conv6, conv7])

    conv8 = Conv1DTranspose(filters=16, kernel_size=256, padding='same', activation='tanh')(conv7)
    conv8 = BatchNormalization()(conv8)
    #skip connection
    conv8 = Add()([conv5, conv8])

    conv9 = Conv1DTranspose(filters=32, kernel_size=128, padding='same', activation='tanh')(conv8)
    conv9 = BatchNormalization()(conv9)
    #skip connection
    conv9 = Add()([conv4, conv9])

    conv10 = Conv1DTranspose(filters=64, kernel_size=64, padding='same', activation='tanh')(conv9)
    conv10 = BatchNormalization()(conv10)
    #skip connection
    conv10 = Add()([conv3, conv10])

    conv11 = Conv1DTranspose(filters=128, kernel_size=32, padding='same', activation='tanh')(conv10)
    conv11 = BatchNormalization()(conv11)
    #skip connection
    conv11 = Add()([conv2, conv11])

    conv12 = Conv1DTranspose(filters=256, kernel_size=16, padding='same', activation='tanh')(conv11)
    conv12 = BatchNormalization()(conv12)
    #skip connection
    conv12 = Add()([conv1, conv12])

    # output layer - wiht or without tanh activation?
    output_layer = Dense(1)(conv12)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v812')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model


def build_model_HiFiGAN_v811(input_shape, output_shape, config):
    """model with autoencoder structure and more skip connections"""

    # add skip connections from input to output
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    # add 12 1d conv layers with skip connections - encoder -decoder structure  

    # encoder
    conv1 = Conv1D(filters=256, kernel_size=16, padding='causal', activation='tanh')(input_layer)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv1D(filters=128, kernel_size=32, padding='causal', activation='tanh')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=64, padding='causal', activation='tanh')(conv2)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv1D(filters=32, kernel_size=128, padding='causal', activation='tanh')(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=16, kernel_size=256, padding='causal', activation='tanh')(conv4)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv1D(filters=8, kernel_size=512, padding='causal', activation='tanh')(conv5)
    conv6 = BatchNormalization()(conv6)

    # decoder
    conv7 = Conv1D(filters=8, kernel_size=512, padding='causal', activation='tanh')(conv6)
    conv7 = BatchNormalization()(conv7)
    #skip connection
    conv7 = Add()([conv6, conv7])

    conv8 = Conv1D(filters=16, kernel_size=256, padding='causal', activation='tanh')(conv7)
    conv8 = BatchNormalization()(conv8)
    #skip connection
    conv8 = Add()([conv5, conv8])

    conv9 = Conv1D(filters=32, kernel_size=128, padding='causal', activation='tanh')(conv8)
    conv9 = BatchNormalization()(conv9)
    #skip connection
    conv9 = Add()([conv4, conv9])

    conv10 = Conv1D(filters=64, kernel_size=64, padding='causal', activation='tanh')(conv9)
    conv10 = BatchNormalization()(conv10)
    #skip connection
    conv10 = Add()([conv3, conv10])

    conv11 = Conv1D(filters=128, kernel_size=32, padding='causal', activation='tanh')(conv10)
    conv11 = BatchNormalization()(conv11)
    #skip connection
    conv11 = Add()([conv2, conv11])

    conv12 = Conv1D(filters=256, kernel_size=16, padding='causal', activation='tanh')(conv11)
    conv12 = BatchNormalization()(conv12)
    #skip connection
    conv12 = Add()([conv1, conv12])

    # output layer
    output_layer = Dense(1)(conv12)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='HiFi-GAN_v811')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model



def build_model_CNN_v9(input_shape, output_shape, config):
    """CNN model with 128 filters and 32 kernel size
    with skip connections"""

    # input shape # (132300, 1)
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    filter_size = 128

    # add 12 1d conv layers with skip connections

    conv1 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(input_layer)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv2)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv4)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv5)
    conv6 = BatchNormalization()(conv6)

    conv7 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv6)
    conv7 = BatchNormalization()(conv7)
    #skip connection
    conv7 = Add()([conv6, conv7])

    conv8 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv7)
    conv8 = BatchNormalization()(conv8)
    #skip connection
    conv8 = Add()([conv5, conv8])

    conv9 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv8)
    conv9 = BatchNormalization()(conv9)
    #skip connection
    conv9 = Add()([conv4, conv9])

    conv10 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv9)
    conv10 = BatchNormalization()(conv10)
    #skip connection
    conv10 = Add()([conv3, conv10])

    conv11 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv10)
    conv11 = BatchNormalization()(conv11)
    #skip connection
    conv11 = Add()([conv2, conv11])

    conv12 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv11)
    conv12 = BatchNormalization()(conv12)
    #skip connection
    conv12 = Add()([conv1, conv12])

    # output layer
    output_layer = Dense(1)(conv12)
    #tanh activation
    output_layer = keras.layers.Activation('tanh')(output_layer)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='CNN_v9')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model


def build_model_CNN_v91(input_shape, output_shape, config):
    """CNN model with 128 filters and 32 kernel size
    with one skip connection"""

    # input shape # (132300, 1)
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    filter_size = 128

    # add 12 1d conv layers with skip connection

    conv1 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(input_layer)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv2)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv4)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv5)
    conv6 = BatchNormalization()(conv6)


    conv7 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv6)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv7)
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv8)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv9)
    conv10 = BatchNormalization()(conv10)

    conv11 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv10)
    conv11 = BatchNormalization()(conv11)

    conv12 = Conv1D(filters=filter_size, kernel_size=32, padding='causal', activation='tanh')(conv11)
    conv12 = BatchNormalization()(conv12)

    #skip connection
    conv12 = Add()([conv1, conv12])

    # output layer
    output_layer = Dense(1)(conv12)
    #tanh activation
    output_layer = keras.layers.Activation('tanh')(output_layer)


    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='CNN_v91')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model



def autoencoder(input_shape, output_shape, config):
    '''
    Implementation of autoencoder, adjusted for speech enhancement
    
    References
    ----------
    [1] Alexandre Defossez, Gabriel Synnaeve, Yossi Adi (2020). Real Time Speech Enhancement in the Waveform Domain.
        Interspeech 2020. https://doi.org/10.48550/arXiv.2006.12847
    
    '''
    # Define list for skip connections
    skip_connections = []

    # input layer
    input = Input(shape = input_shape)

    filter = 8
    depth = 5
    kernel_size = 8
    # biggest filter size = filter * 2**(depth-1)

    x  = Conv1D(filters=1, kernel_size=1)(input)
    # Encoder path
    for i in range(depth):
        x = Conv1D(filters=filter * 2**i, kernel_size=kernel_size, padding='same')(x)
        x = keras.layers.Activation('tanh')(x)
        x = BatchNormalization()(x)
        skip_connections.append(x)
    
    # BLSTM Bottleneck
    x = LSTM(filter * 2**(depth-1), return_sequences=True, return_state=False)(x)
    x = LSTM(filter * 2**(depth-1), return_sequences=True, return_state=False)(x)

    # Decoder Path
    for i in range(depth):
        x = Conv1DTranspose(filters=filter * 2**(depth-1-i), kernel_size=kernel_size, padding='same')(x)
        x = keras.layers.Activation('tanh')(x)
        x = BatchNormalization()(x)
        skip = skip_connections.pop(-1)
        x = Add()([x, skip])
    
    # output layer
    output_layer = Dense(1)(x)

    model = Model(inputs=input, outputs=output_layer, name='autoencoder')
    return model



def build_model_CNN_v92(input_shape, output_shape, config):
    """CNN model with 128 starting filters (encoder-decoder style) and 32 kernel size
    with one skip connection"""

    # input shape # (132300, 1)
    # input layer # (Batchsize, 132300, 1)
    input_layer = Input(shape=input_shape, name='input_layer')

    filter_size = 256

    # add 12 1d conv layers with skip connection

    conv1 = Conv1D(filters=256, kernel_size=32, padding='causal', activation='tanh')(input_layer)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv1D(filters=128, kernel_size=32, padding='causal', activation='tanh')(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=32, padding='causal', activation='tanh')(conv2)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv1D(filters=32, kernel_size=32, padding='causal', activation='tanh')(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv1D(filters=16, kernel_size=32, padding='causal', activation='tanh')(conv4)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv1D(filters=8, kernel_size=32, padding='causal', activation='tanh')(conv5)
    conv6 = BatchNormalization()(conv6)


    conv7 = Conv1D(filters=8, kernel_size=32, padding='causal', activation='tanh')(conv6)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv1D(filters=16, kernel_size=32, padding='causal', activation='tanh')(conv7)
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv1D(filters=32, kernel_size=32, padding='causal', activation='tanh')(conv8)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv1D(filters=64, kernel_size=32, padding='causal', activation='tanh')(conv9)
    conv10 = BatchNormalization()(conv10)

    conv11 = Conv1D(filters=128, kernel_size=32, padding='causal', activation='tanh')(conv10)
    conv11 = BatchNormalization()(conv11)

    conv12 = Conv1D(filters=256, kernel_size=32, padding='causal', activation='tanh')(conv11)
    conv12 = BatchNormalization()(conv12)

    #skip connection
    conv12 = Add()([conv1, conv12])

    # output layer
    output_layer = Dense(1)(conv12)
    #tanh activation
    output_layer = keras.layers.Activation('tanh')(output_layer)


    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name='CNN_v92')
    # get model output shape
    print(f'model output shape: {model.output_shape}')

    return model




# --------------------- NEW --------------------------


def hifi(input_shape, config):
    """original model from hifi gan paper"""

    input_layer = Input(shape=input_shape, name='input_layer')

    filter_size = config['filter_size']
    kernel = config['kernel']
    padding = config['padding']
    activation_func = config['activation_func']
    skip = config['skip']
    act_output = config['act_output']

    # add 12 1d conv layers
    conv1 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(input_layer)
    conv2 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv1)
    conv3 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv2)
    conv4 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv3)
    conv5 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv4)
    conv6 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv5)
    conv7 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv6)
    conv8 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv7)
    conv9 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv8)
    conv10 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv9)
    conv11 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv10)
    conv12 = Conv1D(filters=filter_size, kernel_size=kernel, padding=padding, activation=activation_func)(conv11)
    
    if skip:
        # add skip connection
        conv12 = Add()([conv1, conv12])

    # add output layer
    output_layer = Dense(1)(conv12)

    if act_output == 1:
        #tanh activation
        output_layer = keras.layers.Activation('tanh')(output_layer)

    # get model
    model = Model(inputs=input_layer, outputs=output_layer, name=f'hifi_f{filter_size}_k{kernel}_{padding}_Ac{activation_func}_skip{skip}_act{act_output}')
    return model


        







# --------------------- Not ready yet --------------------------



def Demucs_v1(input_shape, config):
    '''
    Implementation of Demucs Unet, adjusted for speech enhancement
    
    References
    ----------
    [1] Alexandre Defossez, Gabriel Synnaeve, Yossi Adi (2020). Real Time Speech Enhancement in the Waveform Domain.
        Interspeech 2020. https://doi.org/10.48550/arXiv.2006.12847
    
    [2] Reference Implementation: https://github.com/facebookresearch/denoiser/tree/8f006f4c492b24bcbf8b3df33b9d38520c908c55  
    '''

    kernel = config['demucs_kernel']
    stride = config['demucs_stride']
    depth = config['demucs_depth']
    starting_filter = config['demucs_starting_filter']

    # Define list for skip connections
    skip_connections = []

    # input layer
    input = Input(shape = input_shape)

    # Encoder path
    x = demucs_encoder_block(input, filters=starting_filter * 2**0, kernel_size=kernel, stride=stride)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**1, kernel_size=kernel, stride=stride)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**2, kernel_size=kernel, stride=stride)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**3, kernel_size=kernel, stride=stride)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**4, kernel_size=kernel, stride=stride)
    skip_connections.append(x)

    # BLSTM Bottleneck
    x = demucs_blstm(x, starting_filter * 2**4)

    # Decoder Path
    skip = skip_connections.pop(-1)
    x = Add()([x, skip])
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**4,
                             kernel_size=kernel, 
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**3)

    skip = skip_connections.pop(-1)
    x = Add()([x, skip])
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**3,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**2)

    skip = skip_connections.pop(-1)
    x = Add()([x, skip])
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**2,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**1)

    skip = skip_connections.pop(-1)
    x = Add()([x, skip])
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**1,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**0)

    skip = skip_connections.pop(-1)
    x = Add()([x, skip])
    x = demucs_decoder_block(x,
                             filters=starting_filter *  2**0,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=False,
                             transpose_filters=1)

    model = Model(inputs=input, outputs=x, name='Demucs_v1')
    return model




def Demucs_strided(input_shape, config):
    '''
    v102 
    Implementation of Demucs Unet, adjusted for speech enhancement
    
    References
    ----------
    [1] Alexandre Defossez, Gabriel Synnaeve, Yossi Adi (2020). Real Time Speech Enhancement in the Waveform Domain.
        Interspeech 2020. https://doi.org/10.48550/arXiv.2006.12847
    
    [2] Reference Implementation: https://github.com/facebookresearch/denoiser/tree/8f006f4c492b24bcbf8b3df33b9d38520c908c55  
    '''

    kernel = config['demucs_kernel']
    stride = config['demucs_stride']
    depth = config['demucs_depth']
    starting_filter = config['demucs_starting_filter']

    # Define list for skip connections
    skip_connections = []
    
    # input layer
    input = Input(shape = input_shape, name='input_layer')

    # Original length
    orig_length = input_shape[0]

    # Padding so for all layers, size of the input - kernel_size % stride = 0
    valid_length = length_padded(orig_length, config)
    pad_len = int (valid_length - orig_length)
    padding = tf.constant([[0, 0], [0, pad_len], [0, 0]])
    input = tf.pad(input, paddings=padding)

    # Encoder path
    x = demucs_encoder_block(input, filters=starting_filter * 2**0, kernel_size=kernel, stride=stride)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**1, kernel_size=kernel, stride=stride)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**2, kernel_size=kernel, stride=stride)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**3, kernel_size=kernel, stride=stride)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**4, kernel_size=kernel, stride=stride)
    skip_connections.append(x)


    # BLSTM Bottleneck
    x = demucs_blstm(x, starting_filter * 2**4)


    # Decoder Path
    skip = skip_connections.pop(-1)
    #x = x + skip[:, 0:x.shape[1], :]
    x = Add()([x, skip[:, :x.shape[1], :]])
    #x = Add()([x, skip])
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**4,
                             kernel_size=kernel, 
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**3)

    skip = skip_connections.pop(-1)
    #x = x + skip[:, 0:x.shape[1], :]
    x = Add()([x, skip[:, :x.shape[1], :]])
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**3,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**2)

    skip = skip_connections.pop(-1)
    #x = x + skip[:, 0:x.shape[1], :]
    x = Add()([x, skip[:, :x.shape[1], :]])
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**2,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**1)

    skip = skip_connections.pop(-1)
    #x = x + skip[:, 0:x.shape[1], :]
    x = Add()([x, skip[:, :x.shape[1], :]])
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**1,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**0)

    skip = skip_connections.pop(-1)
    #x = x + skip[:, 0:x.shape[1], :]
    x = Add()([x, skip[:, :x.shape[1], :]])
    x = demucs_decoder_block(x,
                             filters=starting_filter *  2**0,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=False,
                             transpose_filters=1)
    
    x = x[:, :orig_length, :]

    model = Model(inputs=input, outputs=x, name='Demucs_strided')
    return model



def Demucs(input, config):
    '''
    Implementation of Demucs Unet, adjusted for speech enhancement
    
    References
    ----------
    [1] Alexandre Defossez, Gabriel Synnaeve, Yossi Adi (2020). Real Time Speech Enhancement in the Waveform Domain.
        Interspeech 2020. https://doi.org/10.48550/arXiv.2006.12847
    
    [2] Reference Implementation: https://github.com/facebookresearch/denoiser/tree/8f006f4c492b24bcbf8b3df33b9d38520c908c55  
    '''

    kernel = config['demucs_kernel']
    stride = config['demucs_stride']
    depth = config['demucs_depth']
    starting_filter = config['demucs_starting_filter']

    # Original length
    # get length with tf.shape(x)[1]
    orig_length = input.shape[1]

    # Define list for skip connections
    skip_connections = []

    # Normalize input
    std = tf.math.reduce_std(input, axis=1, keepdims=True)
    x = input / (1e-5 + std)

    # Padding so for all layers, size of the input - kernel_size % stride = 0
    valid_length = length_padded(orig_length, config)
    pad_len = int (valid_length - orig_length)
    padding = tf.constant([[0, 0], [0, pad_len], [0, 0]])
    x = tf.pad(x, paddings=padding)

    # Encoder path
    x = demucs_encoder_block(x, filters=starting_filter * 2**0, kernel_size=kernel)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**1, kernel_size=kernel)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**2, kernel_size=kernel)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**3, kernel_size=kernel)
    skip_connections.append(x)

    x = demucs_encoder_block(x, filters=starting_filter * 2**4, kernel_size=kernel)
    skip_connections.append(x)

    # BLSTM Bottleneck
    x = demucs_blstm(x, starting_filter * 2**4)

    # Decoder Path
    skip = skip_connections.pop(-1)
    x = Add()([x, skip])
    #x = x + skip[:, 0:x.shape[1], :]
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**4,
                             kernel_size=kernel, 
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**3)

    skip = skip_connections.pop(-1)
    #x = Add()([x, skip])
    x = x[:, 0:x.shape[1], :] + skip[:, 0:x.shape[1], :]
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**3,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**2)

    skip = skip_connections.pop(-1)
    x = x[:, 0:x.shape[1], :] + skip[:, 0:x.shape[1], :]
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**2,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**1)

    skip = skip_connections.pop(-1)
    x = x[:, 0:x.shape[1]-1, :] + skip[:, 0:x.shape[1], :]
    x = demucs_decoder_block(x,
                             filters=starting_filter * 2**1,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=True,
                             transpose_filters=starting_filter * 2**0)

    skip = skip_connections.pop(-1)
    x = x[:, 0:x.shape[1]-1, :] + skip[:, 0:x.shape[1], :]
    x = demucs_decoder_block(x,
                             filters=starting_filter *  2**0,
                             kernel_size=kernel,
                             stride=stride,
                             with_activation=False,
                             transpose_filters=1)

    x = x[:, :orig_length, :]
    x = x * std

    model = Model(inputs=input, outputs=x, name='Demucs')
    return model




def build_model_autoencoder(input_shape, output_shape, config):
    """autoencoder model - 10"""

    # input layer # (Batchsize, 132300, 1)
    # encoder
    encoder_input = Input(shape=input_shape, name='input_layer')

    conv1 = Conv1D(filters=128, kernel_size=32, padding='causal')(encoder_input)
    conv1 = keras.layers.Activation('tanh')(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=64, padding='causal')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv1D(filters=16, kernel_size=128, padding='causal')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv1D(filters=4, kernel_size=256, padding='causal')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    latent_dim = BatchNormalization()(conv4)

    encoder = tf.keras.Model(encoder_input, latent_dim, name='encoder')

    # decoder
    decoder_input = tf.keras.layers.Input(shape=latent_dim.shape[1:])

    # transpose convolution
    conv5 = Conv1DTranspose(filters=4, kernel_size=256, padding='same')(decoder_input)
    conv5 = keras.layers.Activation('tanh')(conv5)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv1DTranspose(filters=16, kernel_size=128, padding='same')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    conv6 = BatchNormalization()(conv6)

    conv7 = Conv1DTranspose(filters=64, kernel_size=64, padding='same')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv1DTranspose(filters=128, kernel_size=32, padding='same')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    conv8 = BatchNormalization()(conv8)

    output_layer = Dense(1)(conv8)
    decoder = tf.keras.Model(decoder_input, output_layer, name='decoder')

    # autoencoder
    encodings = encoder(encoder_input)
    decodings = decoder(encodings)
    # sequential model
    autoencoder = tf.keras.Model(encoder_input, decodings, name='autoencoder')
    # get model output shape
    print(f'model output shape: {autoencoder.output_shape}')

    # model summaries
    encoder.summary()
    decoder.summary()

    return autoencoder




def build_model_autoencoder_v1(input_shape, output_shape, config):
    """autoencoder model with 1 skip connection"""

    # input layer # (Batchsize, 132300, 1)
    # encoder
    encoder_input = Input(shape=input_shape, name='input_layer')

    conv1 = Conv1D(filters=128, kernel_size=32, padding='same')(encoder_input)
    conv1 = keras.layers.Activation('tanh')(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=64, padding='same')(conv1)
    conv2 = keras.layers.Activation('tanh')(conv2)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv1D(filters=16, kernel_size=128, padding='same')(conv2)
    conv3 = keras.layers.Activation('tanh')(conv3)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv1D(filters=4, kernel_size=256, padding='same')(conv3)
    conv4 = keras.layers.Activation('tanh')(conv4)
    conv4 = BatchNormalization()(conv4)

    latent_dim = tf.keras.layers.Lambda(lambda x: x, name='latent_dimension')(conv4)
    encoder = tf.keras.Model(encoder_input, latent_dim, name='encoder')

    # decoder
    decoder_input = tf.keras.layers.Input(shape=latent_dim.shape[1:])

    conv5 = Conv1D(filters=4, kernel_size=256, padding='causal')(decoder_input)
    conv5 = keras.layers.Activation('tanh')(conv5)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv1D(filters=16, kernel_size=128, padding='causal')(conv5)
    conv6 = keras.layers.Activation('tanh')(conv6)
    conv6 = BatchNormalization()(conv6)

    conv7 = Conv1D(filters=64, kernel_size=64, padding='causal')(conv6)
    conv7 = keras.layers.Activation('tanh')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv1D(filters=128, kernel_size=32, padding='causal')(conv7)
    conv8 = keras.layers.Activation('tanh')(conv8)
    conv8 = BatchNormalization()(conv8)

    # skip connection
    skip = Add()([conv8, conv1])

    output_layer = Conv1D(filters=1, kernel_size=32, padding='causal')(conv8)
    decoder = tf.keras.Model(decoder_input, output_layer, name='decoder')

    # autoencoder
    encodings = encoder(encoder_input)
    decodings = decoder(encodings)
    autoencoder = tf.keras.Model(encoder_input, decodings, name='autoencoder_v1')
    # get model output shape
    print(f'model output shape: {autoencoder.output_shape}')

    # model summaries
    encoder.summary()
    decoder.summary()

    return autoencoder

