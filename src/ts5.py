import numpy as np
from rdm import hlp
from utl import lpgz
from rdm.trainer import Trainer
from theano import tensor as T
from utl import hist
from rdm.trainer import R1, R2, CE, L1, L2
from pdb import set_trace
from utl import hist


def flvl(lvl=2, a=20):
    def f(x):
        l = lvl - 1
        t = x - T.arange(0, l, dtype='float32').reshape((l, 1, 1))
        i = T.nnet.sigmoid(a * t)
        y = T.sum(i, 0) / l * (1 - 1e-30) + 5e-31
        return y

    return f


def ts5():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    p1 = np.array(lpgz('../dat/p1.pgz'), dtype='<f4')
    p5 = np.array(lpgz('../dat/p5.pgz'), dtype='<f4')

    dm = p1.shape[-1] * np.power(2.0, [0, 1, 0, -1])
    dm = np.array(dm, 'i4')

    from rdm.sae import SAE
    from copy import deepcopy

    a1 = SAE.from_dim(dm)
    t1 = Trainer(a1, p1, p1, err=CE, reg=R1, lmd=.0, lrt=0.001, bsz=20)

    a2 = deepcopy(a1)
    h2 = hlp.S(1.0)
    a2[-1].s = flvl(3, h2)
    t2 = Trainer(a2, p1, p1, err=CE, reg=R1, lmd=.0, lrt=0.001, bsz=20)

    return a1, a2, p1, p5, h2, t1, t2
