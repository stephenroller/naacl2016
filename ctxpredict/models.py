#!/usr/bin/env python
import numpy as np
import theano.tensor as T
from theano.sandbox.cuda.blas import batched_dot

from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Permute
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.objectives import mse

from layers import FixedEmbedding, Dense2D, Dense3D


def _mse_loss_factory(embedding_var):
    def _mse(y_true, y_pred):
        return mse(embedding_var[y_true].dimshuffle(1, 0, 2), y_pred)
    return _mse

def _cross_categorical(embedding_var):
    def _cc(y_true, y_pred):
        softmaxed = T.nnet.softmax(y_pred.dot(embedding_var.dimshuffle(1, 0)))
        return T.nnet.categorical_crossentropy(softmaxed, y_true[:,0])
    return _cc

def _theano_normalize(matrix):
    return matrix / T.sqrt(T.sum(matrix * matrix, axis=1)).dimshuffle(0, 'x')

def _cosine(embedding_var):
    def _cosine_obj(y_true, y_pred):
        left_normed = _theano_normalize(embedding_var[y_true[:,0]])
        right_normed = _theano_normalize(y_pred)
        return -T.sum(left_normed * right_normed, axis=1)
    return _cosine_obj


class FakeModel(object):
    def __init__(self, space, data_iterator):
        self.space = space
        self.unigram = data_iterator.cbr.unigram
        from collections import Counter
        self.counts = Counter()
        self.counts.update([0])

    def evaluate(self, X, Y, verbose=False):
        predictions = self.predict(X)
        return np.mean(1 == Y)

    def predict(self, X):

        guess = self.counts.most_common(1)[0][0]
        return self.space.matrix[np.array([guess] * len(X))]
        #return self.space.matrix[X[:,1]][:,0,:]
        return self.space.matrix[np.array([1] * len(X))]

    def train_on_batch(self, X, Y):
        self.counts.update(Y.T[0])
        return 0.0

def get_model(modelname, space, R, H, learningrate=.01):
    #opti = optimizers.Adagrad(lr=learningrate)
    opti = optimizers.Adadelta()
    #opti = optimizers.SGD()
    # size of vocab and dimensionality of embeddings
    V, D = space.matrix.shape

    model = Sequential()
    # todo: add an option for letting embeddings vary
    embedding_layer = FixedEmbedding(space.matrix)
    embedding_var = embedding_layer.W
    model.add(embedding_layer)

    try:
        factory = _model_factories[modelname]
        model = factory(model, V, R, D, H)
    except KeyError:
        raise ValueError, "%s is not a valid model name" % modelname

    model.compile(loss=_cosine(embedding_var), optimizer=opti, theano_mode='FAST_RUN', y_shape=(256, 1), y_type='int32')
    #model.compile(loss=_mse_loss_factory(embedding_var), optimizer=opti, theano_mode='FAST_RUN', y_shape=example_Y.shape, y_type='int32')
    #model.compile(loss=_cross_categorical(embedding_var), optimizer=opti, theano_mode='FAST_RUN', y_shape=example_Y.shape, y_type='int32')
    return model

def _make_3d(model, V, R, D, H):
    model.add(Dense3D((R, D), H, activation='tanh'))
    model.add(Flatten())
    model.add(BatchNormalization((R*H,)))
    model.add(Dense(R*H, D, activation='tanh'))
    model.add(Dense(D, D, activation='tanh'))
    model.add(Dense(D, D, activation='linear'))
    return model

def _make_t2d(model, V, R, D, H):
    model.add(Permute((2, 1)))
    model.add(Dense2D((D, R), (D, H), activation='tanh'))
    model.add(Flatten())
    model.add(BatchNormalization((D*H,)))
    model.add(Dense(D*H, D, activation='tanh'))
    model.add(Dense(D, D, activation='tanh'))
    model.add(Dense(D, D, activation='linear'))
    return model

def _make_2d(model, V, R, D, H):
    model.add(Dense2D((R, D), (R, H), activation='tanh'))
    model.add(Flatten())
    model.add(BatchNormalization((R*H,)))
    model.add(Dense(R*H, D, activation='tanh'))
    model.add(Dense(D, D, activation='tanh'))
    model.add(Dense(D, D, activation='linear'))
    return model

def _make_3d2d(model, V, R, D, H):
    model.add(Dense3D((R, D), H, activation='tanh'))
    model.add(BatchNormalization((R, H)))
    model.add(Permute((2, 1)))
    model.add(Dense2D((H, R), (D, H), activation='tanh'))
    model.add(Flatten())
    model.add(BatchNormalization((D*H,)))
    model.add(Dense(D*H, D, activation='tanh'))
    model.add(Dense(D, D, activation='tanh'))
    model.add(Dense(D, D, activation='linear'))
    return model

_model_factories = {
    #'t3d': _make_t3d,
    't2d': _make_t2d,
    #'t3d2d': _make_t3d2d'
    '3d': _make_3d,
    '2d': _make_2d,
    '3d2d': _make_3d2d,
}

MODEL_TYPES = _model_factories.keys()


