#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T

def ndim_tensor(ndim, dtype):
    if ndim == 2:
        return T.matrix(dtype=dtype)
    elif ndim == 3:
        return T.tensor3(dtype=dtype)
    elif ndim == 4:
        return T.tensor4(dtype=dtype)
    return T.matrix(dtype=dtype)

def debug_feedforward(model, X):
    layer_outputs = []
    inpt = X
    for layer in model.layers:
        oldprev = getattr(layer, 'previous', None)
        if oldprev:
            delattr(layer, 'previous')
        oldinput = getattr(layer, 'input', None)
        newinput = ndim_tensor(len(inpt.shape), inpt.dtype)

        setattr(layer, 'input', newinput)
        f = layer.get_output(True)
        fnc = theano.function([newinput], f, on_unused_input='ignore')
        out = fnc(inpt)
        if oldinput:
            setattr(layer, 'input', oldinput)
        else:
            delattr(layer, 'input')
        if oldprev:
            setattr(layer, 'previous', oldprev)
        layer_outputs.append(out)
        inpt = out
    return layer_outputs

def debug_feedforward_pprint(model, X):
    outstr = []
    layer_outputs = debug_feedforward(model, X)
    outstr.append(
        "%20s  %20s  %8s  %8s  %4s  %4s  %8s  %8s" %
            tuple("layer shape outmax outmin sat0 sat1 Wmax Wmin".split()))
    for layer, output in zip(model.layers, layer_outputs):
        satdenom = float(np.prod(output.shape))
        Wmax = hasattr(layer, 'W') and layer.W.get_value().max() or 0
        Wmin = hasattr(layer, 'W') and layer.W.get_value().min() or 0
        outstr.append(
            "%20s  %20s  %8.5f  %8.5f  %4.2f  %4.2f  %8.5f  %8.5f" % (
                layer.__class__.__name__,
                output.shape,
                output.max(), output.min(),
                np.sum(output < .05) / satdenom,
                np.sum(output > .95) / satdenom,
                Wmax, Wmin))
    return "\n".join(outstr)


