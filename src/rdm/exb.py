# expression builders.
import numpy as np
from theano import tensor as T


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
