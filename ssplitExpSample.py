import numpy as np
np.random.seed(1232)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Convolution1D
from keras.utils import np_utils
# from keras import backend as K 
import keras.backend.tensorflow_backend as K

import json

#with K.tf.device('/gpu:1'):
#    K._set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))

# Parameters
## Embedding
vocabulary = 10000
embDimention = 32
embInputLength = 10000

## Convolution1D first
convFirstOutch = 128
convFirstWidth = 5

## Convolution1D second
convSecondOutch = 3
convSecondWidth = 1

# Model construction
model = Sequential()
model.add(Embedding(vocabulary, embDimention, input_length=embInputLength))
model.add(Convolution1D(convFirstOutch, convFirstWidth, border_mode='same', input_shape=(embInputLength, embDimention)))
model.add(Activation('relu'))
model.add(Convolution1D(convSecondOutch, convSecondWidth, border_mode='same', input_shape=(embInputLength, convFirstOutch)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Preprocessing data
y_train = np.random.randint(convSecondOutch, size=(1000, embInputLength))
X_train = y_train

y_train = y_train.reshape(1000 * embInputLength)
Y_train = np_utils.to_categorical(y_train, convSecondOutch).reshape(1000, embInputLength, convSecondOutch)

# Training
model.save('./ssplit_sample_before.h5')

model.fit(X_train, Y_train, batch_size=10, nb_epoch=100, verbose=1)

model.save('./ssplit_sample_after.h5')
