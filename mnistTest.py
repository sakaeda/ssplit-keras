from keras.models import Sequential, load_model
from keras.datasets import mnist
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from keras import backend as K 
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Flatten(input_shape = input_shape))
model.add(Dense(128, input_dim=28*28))
model.add(Activation('relu'))
model.add(Dense(10, input_dim=128))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=100,nb_epoch=10, verbose=1, validation_data=(X_test, Y_test))
