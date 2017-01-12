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

# sample
#corpusF = open('./data/sample/jpnCorpus.json','r')
#lookupF = open('./data/sample/jpnLookup.json','r')
jsonCorpus = json.loads(corpusF.read())
jsonLookup = json.loads(lookupF.read())
corpusF.close()
lookupF.close()

x_train= []
y_train = []
for sent in jsonCorpus['_articles']:
    x_train.append([jsonLookup['_lookup']['_key2id'][list(d.keys())[0]] for d in sent])
    y_train.append([list(d.values())[0] for d in sent])
X_train = np.asarray(x_train)
input_shape = X_train.shape

y_train = np.asarray(y_train)
y_train = y_train.reshape(np.product(y_train.shape))
Y_train = np_utils.to_categorical(y_train, 3).reshape(input_shape[0], input_shape[1], 3)

# Training

#model.fit(X_train, Y_train, batch_size=100, nb_epoch=20, verbose=1)
model.fit(X_train, Y_train, validation_split=0.1, nb_epoch=100, verbose=1)

model.save('./models/ssplit_model.h5')
