import numpy as np
from rdm import hlp
from rdm.cat import Cat
from rdm.pcp import Pcp
from rdm.pcp_odr import PcpOdr as Odr
from rdm.trainer import Trainer
from theano import tensor as T
from utl import lpgz, spgz, hist
from rdm.trainer import R1, R2, CE, L1, L2
from pdb import set_trace


def get_dat1(m=256, f='../raw/wgs/03.vcf.gz'):
    """ get dosage data """
    from gsq.vsq import DsgVsq
    from random import randint
    pos = randint(0, 10000000)
    # raw dosage value in {0, 1, 2}
    itr = DsgVsq(f, bp0=pos, wnd=m, dsg='011')
    dat = next(itr)

    idx = np.random.permutation(dat.shape[0])
    dat = dat[idx, ]
    return np.array(dat, 'i4')


def get_dat2(m=256, f='../raw/wgs/03.vcf.gz'):
    """ get dosage data """
    from gsq.vsq import DsgVsq
    from random import randint
    pos = randint(0, 10000000)
    # raw dosage value in {0, 1, 2}
    itr = DsgVsq(f, bp0=pos, wnd=m, dsg='012')
    dat = next(itr)

    idx = np.random.permutation(dat.shape[0])
    dat = dat[idx, ]
    return np.array(dat, 'i4') / 2


def get_dat3(m=256, f='../raw/wgs/03.vcf.gz'):
    """ get dosage data """
    from gsq.vsq import DsgVsq
    from random import randint
    pos = randint(0, 18000000)

    itr = DsgVsq(f, bp0=pos, wnd=m, dsg='012')
    dat = next(itr)

    idx = np.random.permutation(dat.shape[0])
    dat = dat[idx, ]

    ret = np.ndarray((3, dat.shape[0], dat.shape[1]), dtype='f4')
    for i in np.arange(3):
        ret[i, ] = dat == i
    return ret


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


def data_thr():
    d = get_dat3(m=256)
    return (d[:, :1750, :], d[:, 1750:, :])


def lvl_out(x, lvl=2, a=20):
    from theano import tensor as T
    l = lvl - 1
    t = x - T.arange(0, l, dtype='float32').reshape((l, 1, 1))
    i = T.nnet.sigmoid(a * t)
    y = T.sum(i, 0) / l
    return y


def flvl(lvl=2, a=20):
    def f(x): return lvl_out(x, lvl, a)
    return f


def tst4():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    # {0, .5, 1}
    d1 = lpgz('../dat/d3.pgz') * np.array([0, .5, 1]).reshape(3, 1, 1).sum(0)
    d5 = lpgz('../dat/d7.pgz') * np.array([0, .5, 1]).reshape(3, 1, 1).sum(0)

    dm = d1.shape[-1] * np.power(2.0, [0, 1])
    dm = np.array(dm, 'i4')

    from rdm.sae import SAE
    a1 = SAE.from_dim(dm)
    t1 = Trainer(a1, d1, d1, err=CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    from copy import deepcopy
    a2 = deepcopy(a1)
    t2 = Trainer(a2, d1, d1, err=CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)
    # a2[-1].s = flvl(2, 2.0)

    return a1, a2, t1, t2, d1, d5


def tst5():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    d3 = lpgz('../dat/d3.pgz')  # 3D indicators
    d7 = lpgz('../dat/d7.pgz')  # 3D indicators

    # {0, .5, 1}
    d1 = (d3 * np.array([0, .5, 1]).reshape(3, 1, 1)).sum(0)
    d5 = (d7 * np.array([0, .5, 1]).reshape(3, 1, 1)).sum(0)

    dm = d3.shape[-1] * np.power(2.0, [0, 1])
    dm = np.array(dm, 'i4')

    from rdm.sae import SAE
    a1 = SAE.from_dim(dm)
    t1 = Trainer(a1, d1, d1, err=CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    from copy import deepcopy
    a2 = deepcopy(a1)
    a2[+0] = Odr((dm[0], dm[1]), d3.shape[0], 0, w=a1[+0].w)
    a2[-1] = Odr((dm[1], dm[0]), d3.shape[0], 1, w=a1[-1].w)
    t2 = Trainer(a2, d3, d3, err=Odr.CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    return a1, a2, t1, t2, d1, d3, d5, d7


def tst6():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    d1 = lpgz('../dat/d1.pgz')  # 3D indicators
    d5 = lpgz('../dat/d5.pgz')  # 3D indicators

    d2 = np.zeros((2, d1.shape[0], d1.shape[1]))
    d2[0][d1 == 0] = 1
    d2[1][d1 == 1] = 1
    d6 = np.zeros((2, d5.shape[0], d5.shape[1]))
    d6[0][d5 == 0] = 1
    d6[1][d5 == 1] = 1

    dm = d1.shape[-1] * np.power(2.0, [0, 1])
    dm = np.array(dm, 'i4')

    from rdm.sae import SAE
    a1 = SAE.from_dim(dm)
    t1 = Trainer(a1, d1, d1, err=CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    from copy import deepcopy
    a2 = deepcopy(a1)
    a2[+0] = Odr((dm[0], dm[1]), d2.shape[0], 0, w=a1[+0].w)
    a2[-1] = Odr((dm[1], dm[0]), d2.shape[0], 1, w=a1[-1].w)
    t2 = Trainer(a2, d2, d2, err=Odr.CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    return a1, a2, t1, t2, d1, d2, d5, d6


def ts5():
    b = np.repeat([1, 2], 256).reshape(2, 1, 256)
    o = Odr((512, 256), 3, b=b)
    x = np.random.uniform(0, 1, 1750 * 256).reshape(1750, 256)
    w = o.w.eval()
    b = o.b.eval()
    i = x.dot(w) + b
    p = o.s(i).eval()
    return o, x, w, b, i, p


def ts6():
    b = np.repeat([1, 2], 512).reshape(2, 1, 512)
    o = Odr((256, 512), 3, mod=0, b=b)

    x = get_dat3(256)

    w = o.w.eval()
    b = o.b.eval()
    return o, x, w, b
