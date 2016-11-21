import numpy as np
import sys
try:
    sys.path.extend(['..'] if '..' not in sys.path else [])
    from trainer import Trainer as Tnr
    from sae import SAE
    import hlp
except ValueError:
    from .trainer import Trainer as Tnr
    from .sae import SAE
    from . import hlp
import os
from theano import tensor as T
from theano import scan as SC
from theano import function as F


def jac(y, b):
    """ jacobian expression builder. """
    J, U = SC(
        lambda i, y, b: T.grad(y[i], b),
        sequences=T.arange(y.shape[0]),
        non_sequences=[y, b])

    f = F([b], J, updates=U)
    return f


def test():
    x = T.dvector('x')
    y = x ** 2
    J, U = SC(
        lambda i, y, x: T.grad(y[i], x),
        sequences=T.arange(y.shape[0]),
        non_sequences=[y, x])
    f = F([x], J, updates=U)
    return f, U


# for testing purpose
def rdat(fdr='../../raw/W08/00_GNO', seed=None):
    # pick data file
    np.random.seed(seed)
    fnm = np.random.choice(os.listdir(fdr))
    dat = np.load(os.path.join(fdr, fnm))
    gmx = dat['gmx'].astype('f')

    # fix MAF > .5
    __i = np.where(gmx.sum((0, 1)) > gmx.shape[0])[0]
    gmx[:, :, __i] = 1 - gmx[:, :, __i]
    __x = gmx.reshape(gmx.shape[0], -1)

    # set up neral network
    # from exb import Ods
    dim = __x.shape[-1]

    # dm8 = [dim, dim * 8, dim * 4]
    # dm4 = [dim, dim * 4, dim * 2]
    dm2 = [dim, dim * 2]
    # dm1 = [dim, dim, dim]
    # nt8 = SAE.from_dim(dm8)
    # nt4 = SAE.from_dim(dm4)
    nt2 = SAE.from_dim(dm2)
    # nt1 = SAE.from_dim(dm1)
    # nnt[-1].shp = S(1.0, 'Shp')

    # dat = {'__x': __x, 'nt8': nt8, 'nt4': nt4, 'nt2': nt2}
    return {'__x': __x, 'nnt': nt2}


def main():
    d = rdat()
    x, n = d['__x'], d['nnt']
    t1 = Tnr(hlp.paint(n), x, inc=1.04, dec=0.85)
    t2 = Tnr(hlp.paint(n), x, inc=1.00, dec=1.00)
    t2.__onep__ = None
    return t1, t2
