import numpy as np
from rdm import hlp
from rdm.cat import Cat
from rdm.trainer import Trainer
from utl import lpgz, spgz
from rdm.trainer import R1, R2, CE, L1, L2
from pdb import set_trace


def pre_train1(stk, dat, trainers=None, rep=1, nep=5, reg=R1, lmd=.1):
    """ layer-wise pre-train."""
    # the depth
    dp = len(stk) // 2

    # the trainers
    if trainers is None:
        tr = [None] * dp
    else:
        tr = trainers

    # repeatitive pre-training
    for r in range(rep):
        di = dat
        for i in range(dp):
            # wire the encoder to its decoder counterpart
            ec, dc = stk[i], stk[-1 - i]
            ae = Cat(ec, dc)

            # the trainer
            if tr[i] is None:
                tr[i] = Trainer(ae, di, di, reg=reg, lrt=.005, lmd=lmd)
            else:
                tr[i].src.set_value(di)
                tr[i].dst.set_value(di)
            tr[i].lmd.set_value(lmd)
            tr[i].tune(nep)

            # wire the data to the bottom of the tuple, the output
            # on top is the training material for next layer
            di = ec(di).eval()

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


def data_two():
    d0 = get_dat2(m=256)
    d2 = d0[:1750, ]
    d6 = d0[1750:, ]
    return d2, d6


def test_one():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    d1 = lpgz('../dat/d1.pgz')
    d5 = lpgz('../dat/d5.pgz')

    from rdm.ae import AE
    a1 = AE([d1.shape[1], d1.shape[1] * 2])
    t1 = Trainer(a1, d1, d1, err=CE, reg=R1, lmd=.0, lrt=0.001, bsz=1)

    from copy import deepcopy
    # a2 = deepcopy(a1)
    # t2 = Trainer(a2, d1, d1, reg=R1, lmd=.0, lrt=0.001, bsz=d1.shape[0])

    a2 = deepcopy(a1)
    t2 = Trainer(a2, d1, d1, err=L2, reg=R1, lmd=.0, lrt=0.001, bsz=1)

    # from time import time as tm
    # time0 = tm()
    # t1.tune(50)
    # time1 = tm()
    # print(time1 - time0)

    # time0 = tm()
    # t2.tune(50)
    # time1 = tm()
    # print(time1 - time0)

    return a1, t1, a2, t2, d1, d5


def lvl_out(x, lvl=2, a=20):
    from theano import tensor as T
    l = lvl - 1
    t = x - T.arange(0, l, dtype='float32').reshape((l, 1))
    i = T.nnet.sigmoid(a * t)
    y = T.sum(i, 0) / l
    return y


def flvl(lvl=2, a=20):
    def f(x): return lvl_out(x, lvl, a)
    return f


def test_two():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    d2 = lpgz('../dat/d2.pgz')  # {0, .5, 1}
    d6 = lpgz('../dat/d6.pgz')  # {0, .5, 1}

    dm = d2.shape[1] * np.power(2.0, [0, 1, 0, -1, -2, -3, -4])
    dm = np.array(dm, 'i4')

    from rdm.sae import SAE
    a2 = SAE.from_dim(dm)
    a2[-1].s=flvl(3, 5)
    t2 = Trainer(a2, d2, d2, err=CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)
    # L2(a2(d2), d2).eval(); CE(a2(d6), d6).eval(); L2(a2(d6), d6).eval()

    from copy import deepcopy
    a3 = deepcopy(a2)
    t3 = Trainer(a3, d2, d2, err=L2, reg=R1, lmd=.0, lrt=0.005, bsz=10)
    # CE(a3(d2), d2).eval(); L2(a3(d6), d6).eval(); CE(a3(d6), d6).eval()

    return a2, t2, a3, t3, d2, d6


def test_superfit(dat):
    hlp.set_seed(None)

    nnt = get_SDA([256, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1])

    print('pre-train:')
    tr1 = None
    nnt, tr1 = pre_train1(nnt, dat, tr1, 20, 30, reg=R1, lmd=.0)

    print('fine tune:')
    tr2 = None
    nnt, tr2 = fine_tune1(nnt, dat, tr2, 300, reg=R1, lmd=.0)

    return nnt, tr1, tr2
