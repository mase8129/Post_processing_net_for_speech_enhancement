# Dependencies
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import os.path
import glob
import json
import numpy as np
import datetime        
import argparse
from helpers_oM import *


# AUTOTUNE for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

#--------------------------------------------

def main():

    print('Tensorflow version: ' + tf.__version__)
    print(tf.config.list_physical_devices('GPU'))     

    # initialize log_dir
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # make directory if not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    #parser.add_argument('--n_filters', type=int, default=128)
    #parser.add_argument('--kernel_size', type=int, default=32)
    parser.add_argument('--DS', type=int, default=None, help='Number of elements in dataset. If None, then all elements are used.')
    parser.add_argument('--HPC', type=int, default=0, help='If 1, then run on HPC.')
    args = parser.parse_args()
    
    # get args to config-dict
    config = vars(args)
    
    # add values to config
    config['loss_func'] = 'stft'
    config['shuffle_buffer_size'] = 1000
    config['sr'] = 44100
    config['shift_samples'] = int((441/44100) * config['sr']) # 441 samples bei 44100 Hz
    # set paths to tfrecords
    if config['HPC'] == 1:
        #config['train_paths'] = '/beegfs/home/users/m/mase8129/scratch/marius-s/Dataset/train_tfrecords/*.tfrecords'
        #config['test_paths'] = '/beegfs/home/users/m/mase8129/scratch/marius-s/Dataset/test_tfrecords/*.tfrecords'
        config['train_paths'] = '/beegfs/scratch/marius-s/Dataset/train_tfrecords/*.tfrecords'
        config['test_paths'] = '/beegfs/scratch/marius-s/Dataset/test_tfrecords/*.tfrecords'
    else:
        config['train_paths'] = '/Users/marius/Documents/Uni/TU_Berlin_Master/Masterarbeit/Dataset/train_tfrecords/*.tfrecords'
        config['test_paths'] = '/Users/marius/Documents/Uni/TU_Berlin_Master/Masterarbeit/Dataset/test_tfrecords/*.tfrecords'

    print(config)
    

    #--------------------------------------------
    
    # load train tfrecords
    tfrecords_paths = glob.glob(config['train_paths'])
    train_dataset = tf.data.TFRecordDataset(tfrecords_paths)

    # set number of elements in train_dataset from args
    if config['DS'] == None:
        train_dataset = train_dataset
    else:
        train_dataset = train_dataset.take(config['DS'])

    # load and preprocess train_dataset
    train_dataset = train_dataset.map(decode_tf_records, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.map(shift_samples_x, num_parallel_calls=AUTOTUNE)

    # data augmentation
    # sample shift
    # pitch shift
    # time stretch
    # add noise
    #train_dataset = train_dataset.map(lambda x, y:  ,num_parallel_calls=AUTOTUNE)

    # count elements in train_dataset
    print(f'Number of elements in train_dataset: {len([d for d in train_dataset])}')

    # batching and shuffling
    train_dataset = train_dataset.shuffle(config['shuffle_buffer_size']).batch(config['batch_size']).prefetch(buffer_size=AUTOTUNE)

    # get shape of element and network input shape
    for d in train_dataset:
        print(f'Shape of element voicefixer after batching: {d[0].shape}')
        print(f'Shape of element produced after batching: {d[1].shape}')
        # get input_shape and set output_shape
        input_shape = d[0].shape[1:]
        output_shape = input_shape
        print(f'Model input_shape: {input_shape}, Model output_shape: {output_shape}')
        break


    #-----------
    # load test tfrecords
    tfrecords_paths = glob.glob(config['test_paths'])
    test_dataset = tf.data.TFRecordDataset(tfrecords_paths)

        # if config['DS_take'] is not 'all', then convert to int
    if config['DS'] == None:
        test_dataset = test_dataset
    else:
        test_dataset = test_dataset.take(config['DS'])
    
    # load and preprocess test_dataset
    test_dataset = test_dataset.map(decode_tf_records, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(shift_samples_x, num_parallel_calls=AUTOTUNE)

    # count elements in test_dataset
    print(f'Number of elements in test_dataset: {len([d for d in test_dataset])}')
    # batching and shuffling
    test_dataset = test_dataset.shuffle(config['shuffle_buffer_size']).batch(config['batch_size']).prefetch(buffer_size=AUTOTUNE)


    #--------------------------------------------
    # # define models

    # model to map inputs to outputs
    def build_model_small(input_shape, config):
        
        model = keras.Sequential(name='small')
        
        # add input layer
        model.add(keras.Input(shape=input_shape, name='input_layer'))

        # add output layer
        model.add(keras.layers.Dense(output_shape[1], name='output_layer'))
        print(f'model output shape: {model.output_shape}')
    
        return model



    # 4layer model 
    def build_model_mid(input_shape, config):
    
        model = keras.Sequential(name='4layer')

        # add input layer
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
    def build_model_HiFiGAN(input_shape, config):
    
        # # define dilation rates with max of 254 from WAVENET
        # dilation_rates = [2**i if i<8 else 254 for i in range(config['n_layers'])]
        # print(f'dilation_rates: {dilation_rates}')
    

        # define model
        model = keras.Sequential(name='HiFi-GAN_PP-Net')
        model.add(keras.Input(shape=input_shape, name='input_layer'))
    
        # add layer 
        model.add(keras.layers.Conv1D(filters=128, kernel_size=32, padding='same'))
        model.add(keras.layers.Activation('tanh'))
    
        # Add Conv1D layers
        for _ in range(12):
            model.add(keras.layers.Conv1D(filters=128, kernel_size=32, padding='same'))
            model.add(keras.layers.Activation('tanh'))
    
        # Add the final Conv1D layer without activation layer
        model.add(keras.layers.Dense(output_shape[1], name='output_layer'))
        print(f'model output shape: {model.output_shape}')
    
        return model




    #--------------------------------------------
    # define callbacks

    # LR Scheduler
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    # stop training if val_loss does not decrease
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=30)


    #--------------------------------------------


    # get model
    #model = build_model_small(input_shape, config)
    model = build_model_mid(input_shape, config)
    #model = build_model_PP(input_shape, config)
    #model = build_model_HiFiGAN(input_shape, config)

    #save model name to config
    config['model_name'] = model.name 
    
    # compile model
    model.compile(optimizer = keras.optimizers.legacy.Adam(learning_rate=config['learning_rate']),
                  loss = CustomLoss()
                  )
    
    # print model summary
    print('------------------------------')
    model.summary()    

    #-----------------------------------

    # fit model
    history = model.fit(train_dataset,
                        epochs=config['n_epochs']
                        ,validation_data=test_dataset
                        )
    

    # save model if running on hpc
    if config['HPC'] == 1:
        model.save(log_dir + '/model_' + config['model_name'] + '.keras')        

    # save history
    # change type of elements in history.history to float (for json)
    # history.history['lr'] = [float(i) for i in history.history['lr']]
    with open(log_dir + '/history.json', 'w+') as fp:
        json.dump(history.history, fp, sort_keys=True, indent=4)

    # save config to log_dir
    with open(log_dir + '/config.json', 'w+') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)
    

    
    #--------------------------------------------
    # plot loss and accuracy


    #--------------------------------------------
    # preditct one speechfile from test_dataset

    # get one speechfile from test_dataset
    speech_for_predicition = set_1_speechfile(test_dataset, log_dir, config)
    # predict
    y_pred_tf = model(speech_for_predicition, training=False)



    
    # save audiofile with tensorflow
    y_pred_tf = tf.squeeze(y_pred_tf, axis=-1)
    audio_tf = tf.audio.encode_wav(y_pred_tf, int(config['sr']))
    tf.io.write_file(log_dir + '/_predicted' + '.wav', audio_tf)
   


# call main    
if __name__ == "__main__":
    main()
    

