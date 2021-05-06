import numpy as np
import matplotlib.pyplot as plt

import os
from os.path import join

import librosa

import sklearn
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.layers import Activation, BatchNormalization, Dropout
from keras import regularizers
from keras.optimizers import Adam


datapath = 'F:/MFCC'  # Data path
datapath_weight = 'F:PyCharm/PSA/Weights'

dataset = dict()  # Create an empty dictionary

for folder in os.listdir(datapath):  # Open data path and Read folder
    print(f'Folder "{folder}" is loading.')
    for file in os.listdir(join(datapath, folder)):  # Open folder and Read sound file

        # OriSignal∈R(1*n); join(...): Soundfile; sr: target sampling rate; convert signal to mono; only load up 4s
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
x_test = np.expand_dims(x_test, axis=3)

# Converts a class vector (integers) to binary class matrix
y_test = keras.utils.to_categorical(y_test)


# CNN Model
# ModelC3: Conv2/2/2+GAP+FC+FC lr=0.0001 l2=0.01 dr=0.5
def ModelC3_Prediction(x_test, y_test, weights):

    print(f'Model C3 is predicting')

    model = Sequential(name='ModelC3')

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
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='FC_1'))

    # Dropout
    model.add(Dropout(0.5, name='dropout'))

    # Output layer
    model.add(Dense(18, activation='softmax', name='FC_2'))

    # Optimizer that implements the Adam algorithm
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Configures the model for training
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # load weights
    model.load_weights(join(datapath_weight, weights))

    # Amount of the real Samples
    ar = len(y_test)

    # Prediction
    pred_results = model.predict(x_test[0:ar:1, :, :], batch_size=8)

    # Prediction of Category
    pa = pred_results.argmax(axis=1)  # Category number
    print('Prediction', pa)

    # Actual category
    ya = y_test[0:ar:1, :].argmax(axis=1)
    print('Ture', ya)

    # print the false prediction and the prediction correct rate of sample
    k = 0
    for j in range(len(y_test)):
        if pa[j] != ya[j]:
            print('Nr', j)
            print('Prediction', pa[j], 'Ture', ya[j])
            k = k + 1

    Acc = 1 - k / len(y_test[0:ar:1, :])
    print('False', k)
    print('Samples', ar)
    print('Accuracy', Acc)
    print('\n-----------------------------------------------------------------------------------------------------------')


if __name__ == "__main__":

    for weights in os.listdir(datapath_weight):  # Open data path and Read folder
        print(f'Weights "{weights}" is loaded.')

        ModelC3_Prediction(x_test, y_test, weights)