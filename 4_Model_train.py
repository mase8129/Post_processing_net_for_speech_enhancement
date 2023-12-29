# Dependencies
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, Conv2D, UpSampling1D, concatenate, Dense, BatchNormalization, GlobalAveragePooling1D, Flatten

# Helper libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import os.path
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime        
import argparse
from helpers import *
from models import *
import librosa.display
import librosa


# AUTOTUNE for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

#--------------------------------------------

def main():

    print('------------------------------')
    print('Tensorflow version: ' + tf.__version__)
    print("GPU", "available, YES" if tf.config.list_physical_devices("GPU") else "no GPU")

    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--DS', type=int, default=None, help='Number of elements in dataset. If None, then all elements are used.')
    parser.add_argument('--HPC', type=int, default=0, choices=[0, 1], help='If 1, then run on HPC.')
    parser.add_argument('--loss_func', type=str, default='mix2')
    parser.add_argument('--kernel', type=int, default=16, choices=[16, 32])
    parser.add_argument('--sr', type=int, default=22050, choices=[44100, 22050])
    parser.add_argument('--act_output', type=int, default=0)
    args = parser.parse_args()
    
    # get args to config-dict
    config = vars(args)
    
    # add values to config
    if config['DS'] == None:
        config['shuffle_buffer_size'] = 25000
    else:
        config['shuffle_buffer_size'] = config['DS']

    # set paths to tfrecords
    if config['HPC'] == 1:
        config['train_paths'] = '/beegfs/scratch/marius-s/Dataset/train_tfrecords/*.tfrecords'
        config['test_paths'] = '/beegfs/scratch/marius-s/Dataset/test_tfrecords/*.tfrecords'
        config['validation_paths'] = '/beegfs/scratch/marius-s/Dataset/valid_tfrecords/*.tfrecords'

    elif config['HPC'] == 0:
        config['train_paths'] = '/Users/marius/Documents/Uni/TU_Berlin_Master/Masterarbeit/Dataset/train_tfrecords/*.tfrecords'
        config['test_paths'] = '/Users/marius/Documents/Uni/TU_Berlin_Master/Masterarbeit/Dataset/test_tfrecords/*.tfrecords'
        config['validation_paths'] = '/Users/marius/Documents/Uni/TU_Berlin_Master/Masterarbeit/Dataset/valid_tfrecords/*.tfrecords'



    # print config
    print('------------------------------')
    print('Config:')
    print(config)
    print('------------------------------')


    #--------------------------------------------
    # load train tfrecords
    path = config['train_paths']
    train_dataset = load_and_preprocess_dataset(path, config, dset='train')
    # get number of elements in dataset
    print(f'Number of elements in train_dataset: {len([d for d in train_dataset]) * config["batch_size"]}')


    #-----------
    # load test tfrecords
    path = config['test_paths']
    test_dataset = load_and_preprocess_dataset(path, config, dset='test')
    # get number of elements in dataset
    print(f'Number of elements in test_dataset: {len([d for d in test_dataset]) * config["batch_size"]}')

    #-----------
    # load validation tfrecords
    path = config['validation_paths']
    valid_dataset = load_and_preprocess_dataset(path, config, dset='valid')
    # get number of elements in dataset
    print(f'Number of elements in valid_dataset: {len([d for d in valid_dataset]) * config["batch_size"]}')


    # get shape of element and network input shape
    for d in train_dataset:
        print(f'Shape of input after batching: {d[0].shape}')
        print(f'Shape of target after batching: {d[1].shape}')
        # get input_shape and set output_shape
        print(f'Input type of data: {type(d[0])}')     
        input_shape = d[0].shape[1:]
        output_shape = input_shape
        # print shapes
        print(f'Model input_shape: {input_shape}, Model output_shape: {output_shape}')
        break

        
    #--------------------------------------------
    # define callbacks
    # LR Scheduler
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, min_lr=0.0001)
    # stop training if val_loss does not decrease
    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=15)
    callbacks = [reduce_lr]

    #--------------------------------------------

    # set some parameters for model
    config['padding'] = 'causal'
    config['activation_func'] = 'tanh'
    config['skip'] = True
    config['filter_size'] = 128

    # get model
    model = hifi(input_shape, config)


    # compile model
    model.compile(optimizer = keras.optimizers.legacy.Adam(learning_rate=config['learning_rate'], clipnorm=0.5),
                loss = CustomLoss(config)
                )

    #save model name to config
    config['model_name'] = model.name 
    
    # print model summary
    print('------------------------------')
    model.summary()   
    

    #-----------------------------------

    # start timer
    start = datetime.datetime.now()

    # fit model
    history = model.fit(train_dataset,
                        epochs=config['n_epochs']
                        ,validation_data=test_dataset
                        ,callbacks=callbacks
                        )
    
    # stop timer
    stop = datetime.datetime.now()
    time = stop - start
    config['training_time'] = str(time)
    

    # initialize log_dir
    log_dir = make_logdir(config)


    #--------------------------------------------

    # save history
    # change type of elements in history.history to float (for json)
    if reduce_lr in callbacks:
        history.history['lr'] = [float(i) for i in history.history['lr']]
    with open(log_dir + '/history.json', 'w+') as fp:
        json.dump(history.history, fp, sort_keys=True, indent=4)

    # save model if running on hpc
    if config['HPC'] == 1:
        model.save(log_dir + '/model.keras')    

    # save model summary
    with open(log_dir+'/modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))    

    # save config to log_dir
    with open(log_dir + '/config.json', 'w+') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)
    
    
    #--------------------------------------------
    # plot loss
    
    train_loss = history.history['loss']
    eval_loss = history.history['val_loss']
    
    # plot loss and accuracy in one figure
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label='train_loss')
    plt.plot(range(len(eval_loss)), eval_loss, label='eval_loss')
    
    plt.legend()
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(config['model_name'] + ' - ' + config['loss_func'] + ' - ' + str(config['n_epochs']) + ' epochs')
    
    # save plot to disk
    plt.savefig(log_dir + '/_0loss.png')
    plt.close()


    #--------------------------------------------
    # predict, plot and save - input, target and prediction audio

    pred_plot_save(valid_dataset, model, log_dir, config)

   

# call main    
if __name__ == "__main__":
    main()
    

