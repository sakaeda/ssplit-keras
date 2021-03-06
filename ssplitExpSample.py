import numpy as np
np.random.seed(1232)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Convolution1D
from keras.utils import np_utils
# from keras import backend as K 
import keras.backend.tensorflow_backend as K

import json

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

## sample data size
dataSize = 10000

# Model construction
model = Sequential()
model.add(Embedding(vocabulary, embDimention, input_length=embInputLength))
model.add(Convolution1D(convFirstOutch, convFirstWidth, border_mode='same', input_shape=(embInputLength, embDimention)))
model.add(Activation('relu'))
model.add(Convolution1D(convSecondOutch, convSecondWidth, border_mode='same', input_shape=(embInputLength, convFirstOutch)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# load data
corpusF = open('./data/jpnCorpus.json','r')
lookupF = open('./data/jpnLookup.json','r')
jsonCorpus = json.loads(corpusF.read())
jsonLookup = json.loads(lookupF.read())
corpusF.close()
lookupF.close()

x_train = [jsonLookup['_lookup']['_key2id'][list(d.keys())[0]] for d in jsonCorpus['_article']]
X_train = np.asarray(x_train).reshape(1,len(x_train))
y_train = [list(d.values())[0] for d in jsonCorpus['_article']]
y_train = np.asarray(y_train).reshape(len(y_train))
Y_train = np_utils.to_categorical(y_train, 3).reshape(1, 10000, 3)

# Training

#model.fit(X_train, Y_train, batch_size=100, nb_epoch=20, verbose=1)
model.fit(X_train, Y_train, nb_epoch=10, verbose=1)

# model.save('./ssplit_model.h5')
