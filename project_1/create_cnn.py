"""
ECE196 Face Recognition Project
Author: W Chen

Adapted from: https://keras.io/getting-started/functional-api-guide/

Modify this code to write a LeNet with the following requirements:
* Input dimensions: 32x32x1 - Input
* C1: Convolutional Layer - Conv2D
    number of filters: 6
    kernel size: 5x5
    strides: 1 both horizontally and vertically (Set by default.)
    activation function: sigmoid
    output: 6 layers of 28x28 feature maps (Do not need to specify in function.)
* S2: Max Pooling Layer - MaxPooling2D
    pooling size: 2x2
    strides: 2 both horizontally and vertically
    output: 6 layers of 14x14 feature maps (Do not need to specify in function.)
* C3: Convolutional Layer - Conv2D
    number of filters: 16
    kernel size: 5x5
    strides: 1 both horizontally and vertically
    activation function: sigmoid
    output: 16 layers of 10x10 feature maps(Do not need to specify in function.)
* S4: Max Pooling Layer - MaxPooling2D
    pooling size: 2x2
    strides: 2 both horizontally and vertically
    output: 16 layers of 5x5 feature maps (Do not need to specify in function.)
* C5: Convolutional Layer - Conv2D
    number of filters: 120
    kernel size: 5x5
    strides: 1 both horizontally and vertically
    activation function: sigmoid
    output: 120 layers of 1x1 feature maps(Do not need to specify in function.)
* F6: Fully Connected Layer - Dense
    units: 84
    activation function: tanh
    output 84-dimensional vector (This is specified through units.)
* F7: Fully Connected Layer - Dense
    units: 10
    activation function: softmax
    output 10-dimensional vector (This is specified through units.)
"""
# TODO: Import other layers as necessary. (Conv2D, MaxPooling2D)
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model, Sequential

use_sequential = False

if(use_sequential):
    model = Sequential()

    # C1
    model.add(Conv2D(6, kernel_size=(5, 5),
                     strides=(1, 1),
                     activation='sigmoid',
                     input_shape=(32, 32, 1)))

    # S2
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # C3
    model.add(Conv2D(16, kernel_size=(5, 5),
                     strides=(1, 1),
                     activation='sigmoid'))

    # S4
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # C5
    model.add(Conv2D(120, kernel_size=(5, 5),
                     strides=(1, 1),
                     activation='sigmoid'))

    # Flatten the last convolutional layer so it's 1D
    model.add(Flatten())

    # F6
    model.add(Dense(84, activation='tanh'))

    # F7
    model.add(Dense(10, activation='softmax'))

else:
    inputs = Input(shape=(32, 32, 1))

    # C1
    c1 = Conv2D(6, kernel_size=(5, 5),
                strides=(1, 1),
                activation='sigmoid',
                input_shape=(32, 32, 1))(inputs)

    # S2
    s2 = MaxPooling2D(pool_size=(2, 2))(c1)

    # C3
    c3 = Conv2D(16, kernel_size=(5, 5),
                strides=(1, 1),
                activation='sigmoid')(s2)

    # S4
    s4 = MaxPooling2D(pool_size=(2, 2))(c3)

    # C5
    c5 = Conv2D(120, kernel_size=(5, 5),
                strides=(1, 1),
                activation='sigmoid')(s4)

    # Flatten the last convolutional layer so it's 1D

    # F6
    f6 = Dense(84, activation='tanh')(c5)

    # F7
    predictions = Dense(10, activation='softmax')(f6)

    model = Model(inputs=inputs, outputs=predictions)

# Prints model architecture
model.summary()
