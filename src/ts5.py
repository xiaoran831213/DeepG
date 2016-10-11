import numpy as np
from rdm import hlp
from utl import lpgz
from rdm.trainer import Trainer
from theano import tensor as T
from utl import hist
from scipy.stats import norm

from rdm.trainer import R1, R2, CE, L1, L2
from pdb import set_trace
from utl import hist


def pre_train1(stk, dat, tns=None, rep=1, nep=5, **kws):
    """ layer-wise pre-train."""
    # the depth
    dp = len(stk) // 2

    # the trainers
    if tns is None:
        tr = [None] * dp
    else:
        tr = tns

    # repeatitive pre-training
    for r in range(rep):
        di = dat
        for i in range(dp):
            # wire the encoder to its decoder counterpart
            ae = stk.ae[i]

            # the trainer
            if tr[i] is None:
                tr[i] = Trainer(ae, di, di, **kws)
            else:
                tr[i].src.set_value(di)
                tr[i].dst.set_value(di)

            tr[i].tune(nep)

            # wire the data to the bottom of the tuple, the output
            # on top is the training material for next layer
            di = ae.ec(di).eval()

    return stk, tr


def fine_tune1(stk, dat, trainer=None, rep=1, reg=R1, lmd=.1):
    """ Fine tune the entire network."""
    if trainer is None:
        trainer = Trainer(stk, dat, dat, reg=reg, lrt=0.0005, lmd=lmd)
    else:
        trainer.src.set_value(dat)
        trainer.dst.set_value(dat)
    trainer.tune(rep)

    return stk, trainer


def flvl(lv=2, a=20):
    def f(_x):
        tk = np.array(norm.ppf(np.linspace(0, 1, lv + 1))[1:lv], dtype='f4')
        tk = tk.reshape((lv-1, 1, 1))
        mu = _x - tk
        pd = T.nnet.sigmoid(a * mu)
        return T.sum(pd, 0) / (lv - 1) * (1 - 1e-6) + 5e-7
    return f


def ts5():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    p1 = np.array(lpgz('../dat/p1.pgz'), dtype='<f4')
    p5 = np.array(lpgz('../dat/p5.pgz'), dtype='<f4')

    dm = p1.shape[-1] * np.power(2.0, [0, 2, 1, 0, -1, -2, -3, -4, -8])
    dm = np.array(dm, 'i4')

    from rdm.sae import SAE
    from copy import deepcopy

    a0 = SAE.from_dim(dm)

    a1 = deepcopy(a0)
    t1 = pre_train1(stk, dat, trainers=None, rep=1, nep=5, reg=R1)
    t1 = Trainer(
        a1, p1, p1, v_x=p5, v_z=p5, err=CE, reg=R1, lmd=.0, lrt=0.002, bsz=10)

    a2 = deepcopy(a1)
    h2 = hlp.S(1.0)
    a2[-1].s = flvl(3, h2)

    # h3 = hlp.S(1.0)
    # a2[-2].S = flvl(2, h3)
    t2 = Trainer(
        a2, p1, p1, v_x=p5, v_z=p5, err=CE, reg=R1, lmd=.0, lrt=0.002, bsz=10)

    return a1, a2, p1, p5, h2, t1, t2


