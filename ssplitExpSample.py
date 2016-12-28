import numpy as np
np.random.seed(1232)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Convolution1D
from keras.utils import np_utils
from keras import backend as K 

# Parameters
## Embedding
vocalurary = 10000
embDimention = 12
embInputLength = 10000

## Convolution1D first
convFirstOutch = 128
convFirstWidth = 5

## Convolution1D second
convSecondOutch = 3
convSecondWidth = 1

# Model construction
model = Sequential()
model.add(Embedding(vocalurary, embDimention, input_length=embInputLength))
model.add(Convolution1D(convFirstOutch, convFirstWidth, border_mode='same', input_shape=(embInputLength, embDimention)))
model.add(Activation('relu'))
model.add(Convolution1D(convSecondOutch, convSecondWidth, border_mode='same', input_shape=(embInputLength, convFirstOutch)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Preprocessing data
y_train = np.random.randint(3, size=(100, 10000))
X_train = y_train

y_train = y_train.reshape(100*10000)
Y_train = np_utils.to_categorical(y_train, 3).reshape(100, 10000, 3)

# Training
model.fit(X_train, Y_train, batch_size=10, nb_epoch=10, verbose=1)



