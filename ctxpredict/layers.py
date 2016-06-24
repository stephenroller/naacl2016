#!/usr/bin/env python

import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Layer, Dropout, Flatten
from keras.utils.theano_utils import shared_zeros
from keras.initializations import uniform

import theano
import theano.tensor as T
from theano.sandbox.cuda.blas import batched_dot

def glorot_uniform_3d(shape):
    # like glorot uniform, but controls for the fact that
    # there's some independence in our tensor...
    fan_in = shape[1]
    fan_out = shape[2]
    scale = np.sqrt(6. / (fan_in + fan_out))
    #scale = 5e-1
    return uniform(shape, scale)

class FixedEmbedding(Layer):
    def __init__(self, weights):
        super(FixedEmbedding, self).__init__()
        self.input_dim, self.output_dim = weights.shape

        self.input = T.imatrix()
        self.W = shared_zeros((self.input_dim, self.output_dim))
        self.W.set_value(weights)
        self.params = []

    def get_output(self, train=False):
        X = self.get_input(train)
        out = self.W[X]
        return out

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim}




class Dense2D(Layer):
    def __init__(self, input_shape, output_shape, init='glorot_uniform', activation='linear',
                 weights=None):
        super(Dense2D,self).__init__()

        self.init = keras.initializations.get(init)
        self.activation = keras.activations.get(activation)
        input_x, input_y = input_shape
        output_x, output_y = output_shape

        #assert input_y == output_y

        self.input_dim = (input_x, input_y)
        self.latent_dim = (input_y, output_y)
        self.output_dim = (output_x, output_y)

        self.input = T.tensor3()
        self.W = self.init(self.latent_dim)
        self.b = shared_zeros(self.output_dim)

        self.params = [self.W, self.b]

        self.regularizers = []
        self.constraints = []

    def get_output(self, train):
        X = self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__}


class Dense3D(Layer):
    #def __init__(self, input_shape, hidden, init=glorot_uniform_3d, activation='linear',
    def __init__(self, input_shape, hidden, init='glorot_uniform', activation='linear',
                 weights=None):
        super(Dense3D,self).__init__()

        self.init = keras.initializations.get(init)
        self.activation = keras.activations.get(activation)
        input_x, input_y = input_shape
        #output_x, output_y = output_shape

        self.input_dim = (input_x, input_y)
        self.latent_dim = (input_x, hidden, input_y)
        self.output_dim = (input_x, hidden)

        self.input = T.tensor3()
        self.W = self.init(self.latent_dim)
        self.b = shared_zeros(self.output_dim)

        self.params = [self.W, self.b]

        self.regularizers = []
        self.constraints = []

    def get_output(self, train):
        X = self.get_input(train)
        output = self.activation(batched_dot(self.W, X.dimshuffle(1, 2, 0)).dimshuffle(2, 0, 1) +
                                 self.b)
        return output

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "init": self.init.__name__,
                "activation": self.activation.__name__}


if __name__ == '__main__':
    print "Testing 2D"
    model = Sequential()
    model.add(Dense2D((5, 10), (10, 10), activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense2D((10, 10), (10, 10), activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense2D((10, 10), (1, 10), activation='linear'))
    model.add(Flatten())

    base = np.concatenate([np.eye(5), np.zeros((5, 5))], axis=1)
    X = []
    y = []

    for i in xrange(3600):
        idx = np.random.randint(5)
        multiplier = np.eye(10)
        multiplier[idx,idx] = (i%360)/10.
        x = base.dot(multiplier)
        result = x.sum(axis=0)
        X.append(x)
        y.append(np.array(result))

    X = np.array(X)
    y = np.array(y)

    model.compile(loss='mse', optimizer='sgd')
    model.fit(X, y, nb_epoch=10000, batch_size=360)


