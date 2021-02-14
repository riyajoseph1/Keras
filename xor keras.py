# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 00:44:30 2021

@author: josephr9323
"""

'''
Let's start with the simplest example. 
In this quiz you will build a simple multi-layer feedforward neural network to solve the XOR problem.

1. Set the first layer to a Dense() layer with an output width of 8 nodes 
 and the input_dim set to the size of the training samples (in this case 2).
2. Add a tanh activation function.
3. Set the output layer width to 1, since the output has only two classes.
 (We can use 0 for one class and 1 for the other)
4. Use a sigmoid activation function after the output layer.
5. Run the model for 50 epochs.
This should give you an accuracy of 50%. That's ok, but certainly not great.
Out of 4 input points, we're correctly classifying only 2 of them. 
Let's try to change some parameters around to improve. For example,
 you can increase the number of epochs. You'll pass this quiz if you get 75% accuracy.
 Can you reach 100%?

'''

import numpy as np
from keras.utils import np_utils
import tensorflow as tf
# Using TensorFlow 1.0.0; use tf.python_io in later versions
#tf.python_io.control_flow_ops = tf


# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

# One-hot encoding the output
y = np_utils.to_categorical(y)

# Building the model
xor = Sequential()
xor.add(Dense(32, input_dim=2))
xor.add(Activation("tanh"))
xor.add(Dense(2))
xor.add(Activation("sigmoid"))


# Specify loss as "binary_crossentropy", optimizer as "adam",
# and add the accuracy metric
xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])

# Uncomment this line to print the model architecture
xor.summary()

# Fitting the model
history = xor.fit(X, y, epochs=1000, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict(X))