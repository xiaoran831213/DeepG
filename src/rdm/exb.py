# expression builders.
import numpy as np
import theano
from theano import tensor as T
FX = theano.config.floatX


class Ods:
    """ expression builder for ordernal sigmoid expression. """

    def __init__(self, lvl=2, shp=1.0):
        self.shp = shp
        self.lvl = T.constant(lvl, 'float32')

        from scipy.stats import norm
        bar = norm.ppf(np.linspace(0.0, 1.0, lvl + 1))[1:lvl]
        self.bar = bar.reshape(lvl - 1, 1, 1).astype('f')

    def __call__(self, x):
        _ = T.nnet.sigmoid(self.shp * (x - self.bar))
        return T.sum(_, 0) / (self.lvl - 1) * (1 - 1e-6) + 5e-7


def CE(y, z=None):
    """ build symbolic expression of Cross Entrophy
    y: predicted binary probability.
    z: true binary value, either 0 or 1, the default is 0.
    CE = - z * log(y) + (1 - ) * log(1 - y)

    The first dimension denotes the sampling units, and the last
    dimension denotes the value.
    """
    u = -(z * T.log(y) + (1-z) * T.log(1-y)) if z else -T.log(1-y)
    return T.sum((u), -1)


def L2(y, z=None):
    """ build symbolic expression of L2 norm
    y: predicted value
    z: true value, the default is 0.

    L2 = sum((y - z)^2, dim=1:-1)

    The first dimension denote sampling units.
    """
    u = y - z if z else y
    return T.sqrt(T.sum(u**2, -1))


def L1(y, z=None):
    """ build symbolic expression of L1 norm
    y: predicted value
    z: true value, the default is 0.

    The first dimension of y, z denote the batch size.
    """
    u = y - z if z else y
    return T.sum(T.abs_(u), -1)


def L0(y, z=None, thd=1e-06):
    """ build symbolic expression of L0 norm. """
    u = y - z if z else y
    return T.sum((T.abs_(u) > thd), -1, FX)
