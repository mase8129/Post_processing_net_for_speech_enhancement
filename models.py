# imports
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, Conv1DTranspose, UpSampling1D, concatenate, Dense, BatchNormalization, GlobalAveragePooling1D, Flatten, Input, Add
from keras.models import Model
import numpy as np


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





# --------------------- NEW --------------------------




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






# --------------------- Not ready yet --------------------------


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

