import numpy as np
import matplotlib.pyplot as plt

import os
from os.path import join

import librosa

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense
from keras.layers import Activation, BatchNormalization, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


datapath = 'F:/MFCC'  # Data path

dataset = dict()  # Create an empty dictionary

for folder in os.listdir(datapath):  # Open data path and Read folder
    print(f'Folder "{folder}" is loading.')
    for file in os.listdir(join(datapath, folder)):  # Open folder and Read sound file

        # join(...): Soundfile; sr: target sampling rate; convert signal to mono; only load up 4s
        OriSignal, sr = librosa.load(join(join(datapath, folder), file), sr=44100, mono=True, duration=4)

        # Zero padding
        Signal = np.hstack((OriSignal, np.zeros(4 * 44100 - len(OriSignal))))

        # Pre-Emphasis
        eSignal = np.append(Signal[0], Signal[1:] - 0.97 * Signal[:-1])

        # Mel-frequency cepstral coefficients
        MFCC = librosa.feature.mfcc(eSignal, sr=sr, n_fft=1024)

        if dataset:  # dataset not empty
            dataset['samples'] = np.append(dataset['samples'], np.array([MFCC[1:13, :]]), axis=0)
            dataset['labels'] = np.append(dataset['labels'], [os.listdir(datapath).index(folder)])
        else:  # dataset empty
            dataset['samples'] = np.array([MFCC[1:13, :]])
            dataset['labels'] = np.array([os.listdir(datapath).index(folder)])

# Split arrays into random rest and test subsets
x_train, x_test, y_train, y_test = train_test_split(dataset['samples'], dataset['labels'], test_size=0.2, random_state=10, shuffle=True)

# Expand to 4 dimensions
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Converts a class vector (integers) to binary class matrix
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


# CNN Model
# ModelA1: Conv2/2/2+Flatten+FC+FC lr=0.001 l2=0.1 dr=0.5
def ModelA1(x_train, x_test, y_train, y_test):

    print(f'Model A1 is training')

    model = Sequential(name='ModelA1')

    input_shape = (12, 345, 1)

    # 1st. Convolution layers block
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, name='Block1_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same', name='Block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='Block1_pool'))

    # 2nd. Convolution layers block
    model.add(Conv2D(64, (3, 3), padding='same', name='Block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', name='Block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='Block2_pool'))

    # 3rd. Convolution layers block
    model.add(Conv2D(128, (3, 3), padding='same', name='Block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', name='Block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Flatten layer
    model.add(Flatten())

    # Regular densely-connected NN layer
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1), name='FC_1'))

    # Dropout
    model.add(Dropout(0.5, name='dropout'))

    # Output layer
    model.add(Dense(18, activation='softmax', name='FC_2'))

    # Prints a string summary of the NN
    model.summary()

    # Optimizer that implements the Adam algorithm
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Configures the model for training
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Save the best weights
    filepath = 'weights.ModelA1_Best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Trains the model for 500 epochs (iterations on a dataset)
    ModelA1_hist = model.fit(x_train, y_train, batch_size=32, epochs=500, verbose=1, callbacks=checkpoint,
                            validation_data=(x_test, y_test))

    # Plot
    fig = plt.figure(figsize=(15, 5))

    # Plot training_loss & test_loss
    plt.subplot(1, 2, 1)
    plt.plot(ModelA1_hist.history['loss'])
    plt.plot(ModelA1_hist.history['val_loss'])
    plt.title('ModelA1 loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid('on')

    # Plot training_accuracy & test_accuracy
    plt.subplot(1, 2, 2)
    plt.plot(ModelA1_hist.history['accuracy'])
    plt.plot(ModelA1_hist.history['val_accuracy'])
    plt.title('ModelA1 accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid('on')

    plt.show()

    return None

# ModelB1: Conv2/2/2+GAP+FC lr=0.001
def ModelB1(x_train, x_test, y_train, y_test):

    print(f'Model B1 is training')

    model = Sequential(name='ModelB1')

    input_shape = (12, 345, 1)

    # 1st. Convolution layers block
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, name='Block1_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same', name='Block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='Block1_pool'))

    # 2nd. Convolution layers block
    model.add(Conv2D(64, (3, 3), padding='same', name='Block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', name='Block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='Block2_pool'))

    # 3rd. Convolution layers block
    model.add(Conv2D(128, (3, 3), padding='same', name='Block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', name='Block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Global average pooling
    model.add(GlobalAveragePooling2D())

    # Output layer
    model.add(Dense(18, activation='softmax', name='FC'))

    # Prints a string summary of the NN
    model.summary()

    # Optimizer that implements the Adam algorithm
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

     # Configures the model for training
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Save the best weights
    filepath = 'weights.ModelB1_Best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Trains the model for 200 epochs (iterations on a dataset)
    ModelB1_hist = model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=1, callbacks=checkpoint,
                            validation_data=(x_test, y_test))

    # Plot
    fig = plt.figure(figsize=(15, 5))

    # Plot training_loss & test_loss
    plt.subplot(1, 2, 1)
    plt.plot(ModelB1_hist.history['loss'])
    plt.plot(ModelB1_hist.history['val_loss'])
    plt.title('ModelB1 loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid('on')

    # Plot training_accuracy & test_accuracy
    plt.subplot(1, 2, 2)
    plt.plot(ModelB1_hist.history['accuracy'])
    plt.plot(ModelB1_hist.history['val_accuracy'])
    plt.title('ModelB1 accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid('on')

    plt.show()

    return None

# ModelC1: Conv2/2/2+GAP+FC+FC lr=0.001 l2=0.1 dr=0.5
def ModelC1(x_train, x_test, y_train, y_test):

    print(f'Model C1 is training')

    model = Sequential(name='ModelC1')

    input_shape = (12, 345, 1)

    # 1st. Convolution layers block
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, name='Block1_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same', name='Block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='Block1_pool'))

    # 2nd. Convolution layers block
    model.add(Conv2D(64, (3, 3), padding='same', name='Block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', name='Block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='Block2_pool'))

    # 3rd. Convolution layers block
    model.add(Conv2D(128, (3, 3), padding='same', name='Block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', name='Block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Global average pooling
    model.add(GlobalAveragePooling2D())

    # Regular densely-connected NN layer
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1), name='FC_1'))

    # Dropout
    model.add(Dropout(0.5, name='dropout'))

    # Output layer
    model.add(Dense(18, activation='softmax', name='FC_2'))

    # Prints a string summary of the NN
    model.summary()

    # Optimizer that implements the Adam algorithm
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Configures the model for training
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Save the best weights
    filepath = 'weights.ModelC1_Best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Trains the model for 300 epochs (iterations on a dataset)
    ModelC1_hist = model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=1, callbacks=checkpoint,
                            validation_data=(x_test, y_test))

    # Plot
    fig = plt.figure(figsize=(15, 5))

    # Plot training_loss & test_loss
    plt.subplot(1, 2, 1)
    plt.plot(ModelC1_hist.history['loss'])
    plt.plot(ModelC1_hist.history['val_loss'])
    plt.title('ModelC1 loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid('on')

    # Plot training_accuracy & test_accuracy
    plt.subplot(1, 2, 2)
    plt.plot(ModelC1_hist.history['accuracy'])
    plt.plot(ModelC1_hist.history['val_accuracy'])
    plt.title('ModelC1 accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid('on')

    plt.show()

    return None


if __name__ == "__main__":

  i = 0

  # Training 10 Times
  while i < 5:

      # Count
      i = i + 1
      print(f'Nr.{i} Training')

      ModelA1(x_train, x_test, y_train, y_test)
      ModelB1(x_train, x_test, y_train, y_test)
      ModelC1(x_train, x_test, y_train, y_test)