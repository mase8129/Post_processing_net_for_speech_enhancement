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


# AUTOTUNE for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

#--------------------------------------------

def main():

    print('------------------------------')
    print('Tensorflow version: ' + tf.__version__)
    print("GPU", "available, YES" if tf.config.list_physical_devices("GPU") else "no GPU")

    # get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    #parser.add_argument('--n_filters', type=int, default=128)
    #parser.add_argument('--kernel_size', type=int, default=32)
    #parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--DS', type=int, default=None, help='Number of elements in dataset. If None, then all elements are used.')
    parser.add_argument('--HPC', type=int, default=0, help='If 1, then run on HPC.')
    parser.add_argument('--model', type=int, default=0)
    parser.add_argument('--loss_func', type=str, default='stft')
    args = parser.parse_args()
    
    # get args to config-dict
    config = vars(args)
    
    # add values to config
    config['shuffle_buffer_size'] = 1500
    config['sr'] = 44100
    config['shift_samples'] = int((441/44100) * config['sr']) # 441 samples bei 44100 Hz

    # set paths to tfrecords
    if config['HPC'] == 1:
        config['train_paths'] = '/beegfs/scratch/marius-s/Dataset/train_tfrecords/*.tfrecords'
        config['test_paths'] = '/beegfs/scratch/marius-s/Dataset/test_tfrecords/*.tfrecords'
        config['validation_paths'] = '/beegfs/scratch/marius-s/Dataset/valid_tfrecords/*.tfrecords'
    else:
        config['train_paths'] = '/Users/marius/Documents/Uni/TU_Berlin_Master/Masterarbeit/Dataset/train_tfrecords/*.tfrecords'
        config['test_paths'] = '/Users/marius/Documents/Uni/TU_Berlin_Master/Masterarbeit/Dataset/test_tfrecords/*.tfrecords'
        config['validation_paths'] = '/Users/marius/Documents/Uni/TU_Berlin_Master/Masterarbeit/Dataset/valid_tfrecords/*.tfrecords'

    print('------------------------------')
    print('Config:')
    print(config)
    print('------------------------------')


    #--------------------------------------------
    path = config['train_paths']
    train_dataset = load_and_preprocess_dataset(path, config, train_dataset=True)

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
    path = config['test_paths']
    test_dataset = load_and_preprocess_dataset(path, config, train_dataset=False)

    #-----------
    # load validation tfrecords
    path = config['validation_paths']
    valid_dataset = load_and_preprocess_dataset(path, config, train_dataset=False)

    #--------------------------------------------
    # define callbacks

    # LR Scheduler
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    # stop training if val_loss does not decrease
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=15)
    
    callbacks = [reduce_lr, early_stop]

    #--------------------------------------------
    # get model
    if config['model'] == 0:
        model = build_model_small(input_shape, output_shape, config)
    elif config['model'] == 1:
        model = build_model_mid(input_shape, output_shape, config)
    elif config['model'] == 2:
        model = build_model_HiFiGAN(input_shape, output_shape, config)
   
   
    elif config['model'] == 8:
        model = build_model_HiFiGAN_v8(input_shape, output_shape, config)
    elif config['model'] == 81:
        model = build_model_HiFiGAN_v81(input_shape, output_shape, config)
    elif config['model'] == 811:
        model = build_model_HiFiGAN_v811(input_shape, output_shape, config)
    elif config['model'] == 812:
        model = build_model_HiFiGAN_v812(input_shape, output_shape, config)


    elif config['model'] == 9:
        model = build_model_CNN_v9(input_shape, output_shape, config)
    elif config['model'] == 10:
        # define some parameters
        config['demucs_depth'] = 5
        config['demucs_kernel'] = 8
        config['demucs_stride'] = 2
        config['demucs_starting_filter'] = 8
        # get model
        model = Demucs_v1(input_shape, config)

    # elif config['model'] == 9:
    #     # define some parameters
    #     config['demucs_depth'] = 5
    #     config['demucs_kernel'] = 8
    #     config['demucs_stride'] = 2
    #     config['demucs_starting_filter'] = 8
    #     # get model

    #     model_input = Input(shape=input_shape)
    #     model = Demucs(model_input, config)

    # compile model
    model.compile(optimizer = keras.optimizers.legacy.Adam(learning_rate=config['learning_rate']),
                  loss = CustomLoss(config)
                  )
    
    # # GAN model
    # if config['model'] == 6:
    #     Gen, Mel = build_generator(input_shape, output_shape, config)
    #     Disc = build_discriminator(Mel, output_shape, config)
    #     model = keras.models.Sequential([Gen, Disc])

    #     opti = keras.optimizers.legacy.Adam(learning_rate=config['learning_rate'])
    #     Disc.compile(loss="binary_crossentropy", optimizer=opti)
    #     Disc.trainable = False
    #     model.compile(loss="binary_crossentropy", optimizer=opti)

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
    log_dir = make_logdir()

    # save history
    # change type of elements in history.history to float (for json)
    history.history['lr'] = [float(i) for i in history.history['lr']]
    with open(log_dir + '/history.json', 'w+') as fp:
        json.dump(history.history, fp, sort_keys=True, indent=4)

    # save model if running on hpc
    if config['HPC'] == 1:
        model.save(log_dir + '/model.keras')        

    # save config to log_dir
    with open(log_dir + '/config.json', 'w+') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)
    
    
    #--------------------------------------------
    # plot loss
    
    train_loss = history.history['loss']
    eval_loss = history.history['val_loss']
    
    # plot loss and accuracy in one figure
    fig1 = plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label='train_loss')
    plt.plot(range(len(eval_loss)), eval_loss, label='eval_loss')
    
    plt.legend()
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training with ' + str(len(train_loss)) + ' epochs')
    
    # save plot to disk
    plt.savefig(log_dir + '/loss.png')
    plt.close()


    #--------------------------------------------
    # preditct one speechfile from dataset

    # get one speechfile from dataset
    speech_for_predicition = set_1_speechfile(valid_dataset, log_dir, config)

    # get shapes of speech_for_predicition
    print('Shape of speech_for_predicition: ', speech_for_predicition.shape)

    # predict
    y_pred_tf = model.predict(speech_for_predicition)
    # change shape to (len(audio), 1)
    y_pred_np = tf.squeeze(y_pred_tf, axis=-1).numpy()

    # save plot to disk
    plt.figure(figsize=(8, 4))
    x = np.arange(0, len(y_pred_np)/int(config['sr']), 1/int(config['sr']))
    plt.plot(x, y_pred_np)
    plt.title('predicted')
    plt.xlabel('Time in s')
    plt.ylabel('Amplitude')
    plt.savefig(log_dir + '/_predicted.png')
    plt.close()
    
    # save audiofile with tensorflow
    y_pred_tf = tf.squeeze(y_pred_tf, axis=-1)
    audio_tf = tf.audio.encode_wav(y_pred_tf, int(config['sr']))
    tf.io.write_file(log_dir + '/_predicted' + '.wav', audio_tf)
   

# call main    
if __name__ == "__main__":
    main()
    

