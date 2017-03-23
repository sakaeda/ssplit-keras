import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Convolution1D
from keras.utils import np_utils
# from keras import backend as K 
import keras.backend.tensorflow_backend as K

import tensorflow as tf

import json

def gen_train_data(corpus, lookup):
    x_train= []
    y_train = []
    maxLength = -1
    for sents in corpus['_articles']:
        for sent in sents:
            x_train.append([lookup['_lookup']['_key2id'][list(d.keys())[0]] for d in sent])
            y_train.append([list(d.values())[0] for d in sent])
            if len(sent) > maxLength:
                maxLength = len(sent)
    X_train = np.asarray(x_train)
    input_shape = X_train.shape

    print(input_shape)
    print('max length\n')
    print(maxLength)

    y_train = np.asarray(y_train)
    y_train = y_train.reshape(np.product(y_train.shape))
    Y_train = np_utils.to_categorical(y_train, 3).reshape(input_shape[0], input_shape[1], 3)
    return X_train, Y_train

def train(X_train, Y_train, nb_epoch):
    # Parameters
    ## Embedding
    vocabulary = 10000
    embDimention = 128
    embInputLength = 4000

    ## Convolution1D first
    convFirstOutch = 512
    convFirstWidth = 11

    ## Convolution1D second
    convSecondOutch = 128
    convSecondWidth = 7

    ## Convolution1D third
    convThirdOutch = 32
    convThirdWidth = 3

    ## Convolution1D fourth
    convFourthOutch = 3
    convFourthWidth = 1

    config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                allow_growth=True,
                visible_device_list='3'
                )
            )

    sess = tf.Session(config=config)

    K.set_session(sess)

    # Model construction
    model = Sequential()
    model.add(Embedding(vocabulary, embDimention, input_length=embInputLength))
    model.add(
            Convolution1D(
                convFirstOutch,
                convFirstWidth,
                border_mode='same',
                input_shape=(embInputLength, embDimention)
                )
            )
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
            Convolution1D(
                convSecondOutch,
                convSecondWidth,
                border_mode='same',
                input_shape=(embInputLength, convFirstOutch)
                )
            )
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
            Convolution1D(
                convThirdOutch,
                convThirdWidth,
                border_mode='same',
                input_shape=(embInputLength, convSecondOutch)
                )
            )
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(
            Convolution1D(
                convFourthOutch,
                convFourthWidth,
                border_mode='same',
                input_shape=(embInputLength, convSecondOutch)
                )
            )
    model.add(Activation('softmax'))
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy']
            )
    history = model.fit(
            X_train,
            Y_train,
            batch_size=10,
            validation_split=0.1,
            nb_epoch=nb_epoch,
            verbose=1
            )

    return model, history

# load data
bioCorpusF = open('./data/20170130/jpnCorpus.json','r')
lookupF =    open('./data/20170130/jpnLookup.json','r')

# sample
jsonBIOCorpus = json.loads(bioCorpusF.read())
jsonLookup = json.loads(lookupF.read())
bioCorpusF.close()
lookupF.close()

X_train_bio, Y_train_bio = gen_train_data(jsonBIOCorpus, jsonLookup)

# Training
modelBIO, historyBIO = train(X_train_bio, Y_train_bio, 1000)
modelBIO.save('./models/20170321/ssplit_model.h5')
outfBIO =open('./models/20170321/history.json','w')
json.dump(historyBIO.history, outfBIO)
outfBIO.close()
