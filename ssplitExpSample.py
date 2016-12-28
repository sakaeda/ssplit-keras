import numpy as np
np.random.seed(1232)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Convolution1D
from keras.utils import np_utils
from keras import backend as K 

model = Sequential()
model.add(Embedding(10000, 12, input_length=10000))
model.add(Convolution1D(128, 5, border_mode='same', input_shape=(10000, 12)))
model.add(Activation('relu'))
model.add(Convolution1D(3, 1, border_mode='same', input_shape=(10000, 128)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

X_train = np.random.randint(10, size=(10,10000))
y_train = np.random.randint(3, size=(10, 10000))

y_train = y_train.reshape(10*10000)
Y_train = np_utils.to_categorical(y_train, 3).reshape(10, 10000, 3)

model.fit(X_train, Y_train, nb_epoch=10, verbose=1)

