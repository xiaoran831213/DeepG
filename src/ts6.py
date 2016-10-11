import numpy as np
from rdm import hlp
from utl import lpgz
from rdm.trainer import Trainer
from theano import tensor as T
from utl import hist
from scipy.stats import norm

from rdm.trainer import R1, R2, CE, L1, L2
from pdb import set_trace


def pre_train1(stk, tns=None, dat=None, rep=1, nep=1, **kws):
    """ layer-wise pre-train."""

    # the trainers
    if tns is None:
        tns = [None] * len(stk.sa)
    else:
        tns = tns

    # repeatitive pre-training
    for r in range(rep):
        di = dat
        for i, a in enumerate(stk.sa):
            # the trainer
            if tns[i] is None:
                tns[i] = Trainer(a, di, di, **kws)
            if di is not None:
                tns[i].x.set_value(di)
                tns[i].x.set_value(di)

            tns[i].tune(nep)

            # wire the data to the bottom of the tuple, the output
            # on top is the training material for next layer
            di = a.ec(di).eval()

    return tns


def flvl(lv=2, a=20):
    def f(_x):
        tk = np.array(norm.ppf(np.linspace(0, 1, lv + 1))[1:lv], dtype='f4')
        tk = tk.reshape((lv-1, 1, 1))
        mu = _x - tk
        pd = T.nnet.sigmoid(a * mu)
        return T.sum(pd, 0) / (lv - 1) * (1 - 1e-6) + 5e-7
    return f


def ts6():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    p1 = np.array(lpgz('../dat/p1.pgz'), dtype='<f4')
    p5 = np.array(lpgz('../dat/p5.pgz'), dtype='<f4')

    dm = p1.shape[-1] * np.power(
        2.0, [0, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8])
    dm = np.array(dm, 'i4')

    from rdm.sae import SAE
    from copy import deepcopy

    a0 = SAE.from_dim(dm)

    a1 = deepcopy(a0)
    t1 = pre_train1(a1, None, p1, rep=20, reg=R1, lmd=0.001, lrt=0.002, bsz=1)

    a2 = deepcopy(a0)
    h2 = hlp.S(1.0)
    a2[-1].s = flvl(3, h2)
    # h3 = hlp.S(1.0)
    # a2[-2].S = flvl(2, h3)
    t2 = pre_train1(a2, None, p1, rep=20, reg=R1, lmd=0.001, lrt=0.002, bsz=1)

    return a0, a1, a2, p1, p5, h2, t1, t2


def tmp1(a0, a1, a2, p1, p5):
    f0 = Trainer(a0, p1, p1, p5, p5, lrt=0.002, reg=R1, lmd=.001, bsz=1)
    f1 = Trainer(a1, p1, p1, p5, p5, lrt=0.002, reg=R1, lmd=.001, bsz=1)
    f2 = Trainer(a2, p1, p1, p5, p5, lrt=0.002, reg=R1, lmd=.001, bsz=1)
    return f0, f1, f2
