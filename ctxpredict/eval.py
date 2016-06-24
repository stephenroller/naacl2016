#!/usr/bin/env python
import numpy as np

def intrinsic_eval_euclid(model, space, testX, testY):
    retval = np.zeros(testX.shape[0])
    denom = float(space.matrix.shape[0])
    testY = testY[:,0]
    pred = model.predict(testX)
    for i in xrange(testX.shape[0]):
        distances = np.sqrt(np.sum(np.square(space.matrix - pred[i]), axis=1))
        idx = testY[i]
        gold = distances[idx]
        rank = np.sum(distances <= gold) / denom
        retval[i] = rank
    return retval.mean()

from sklearn.preprocessing import normalize
def intrinsic_eval(model, space, testX, testY):
    space = space.normalize()
    retval = np.zeros(testX.shape[0])
    denom = float(space.matrix.shape[0])
    testY = testY[:,0]
    pred = model.predict(testX)
    pred = normalize(pred, norm='l2', axis=1)
    #pred += space.matrix[testY]
    pred = normalize(pred, norm='l2', axis=1)
    dots = pred.dot(space.matrix.T)
    goldscores = np.array([dots[i,yid] for i, yid in enumerate(testY)])
    greaterthan = np.mean(dots >= np.array([goldscores]).T, axis=1)
    return np.mean(greaterthan)

