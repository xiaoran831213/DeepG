import numpy as np
from rdm import hlp
from rdm.cat import Cat
from rdm.pcp import Pcp
from rdm.trainer import Trainer
from utl import lpgz
from rdm.trainer import R1, R2, RN
from pdb import set_trace


def get_data(m=256, f='../raw/wgs/03.vcf.gz'):
    """ get dosage data """
    from gsq.vsq import DsgVsq
    from random import randint
    pos = randint(0, 10000000)
    # raw dosage value in {0, 1, 2}
    itr = DsgVsq(f, bp0=pos, wnd=m, dsg='011')
    dat = next(itr)

    idx = np.random.permutation(dat.shape[0])
    dat = dat[idx, ]
    return np.array(dat, 'f4')


def get_SDA(dm):
    """ create an stacked auto encoder """
    dms = list(zip(dm[:-1], dm[1:], range(len(dm))))
    ecs = [Pcp((i, j), tag='E{}'.format(t)) for i, j, t in dms]
    dcs = [Pcp((j, i), tag='D{}'.format(t)) for i, j, t in dms]

    # constraint weight terms
    for ec, dc in zip(ecs, dcs):
        dc.w = ec.w.T

    # wire them up
    dcs.reverse()
    aes = ecs + dcs
    stk = Cat(*aes)

    return stk


def get_SAE(dm):
    from rdm.sae import SAE
    return SAE.from_dim(dm)


def pre_train1(stk, dat, trainers=None, rep=1, nep=5, reg=R1, lmd=.1):
    """ layer-wise pre-train."""
    # the depth
    dp = len(stk)//2

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
            ec, dc = stk[i], stk[-1-i]
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
    d0 = get_data(m=256)
    d1 = d0[:1750, ]
    d2 = d0[1750:, ]
    return d1, d2

    
def test_one():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    d1 = lpgz('../dat/d1.pgz')
    d5 = lpgz('../dat/d5.pgz')

    from rdm.ae import AE
    a1 = AE([d1.shape[1], d1.shape[1]*2])
    t1 = Trainer(a1, d1, d1, reg=R1, lmd=.0, lrt=0.001, bsz=1)

    from copy import deepcopy
    a2 = deepcopy(a1)
    t2 = Trainer(a2, d1, d1, reg=R1, lmd=.0, lrt=0.001, bsz=d1.shape[0])

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

