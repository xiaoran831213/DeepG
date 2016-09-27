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


def get_dat2(m=256, f='../raw/wgs/03.vcf.gz', d='012'):
    """ get dosage data """
    from gsq.vsq import DsgVsq
    from random import randint
    pos = randint(0, 10000000)

    itr = DsgVsq(f, bp0=pos, wnd=m, dsg=d)
    dat = next(itr)

    idx = np.random.permutation(dat.shape[0])
    dat = dat[idx, ]
    prb = np.array(dat, 'i4') / 2

    bno = np.ndarray((3, dat.shape[0], dat.shape[1]), dtype='f4')
    for i in np.arange(3):
        bno[i, ] = dat == i
    
    return dat, prb, bno


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


def gen_thr():
    d, p, b = get_dat2(m=256)
    d1, d5 = d[..., :1750, :], d[..., 1750:, :]
    p1, p5 = p[..., :1750, :], p[..., 1750:, :]
    b1, b5 = b[..., :1750, :], b[..., 1750:, :]
    spgz('../dat/d1.pgz', d1)
    spgz('../dat/d5.pgz', d5)
    spgz('../dat/p1.pgz', p1)
    spgz('../dat/p5.pgz', p5)
    spgz('../dat/b1.pgz', b1)
    spgz('../dat/b5.pgz', b5)


def lvl_out(x, lvl=2, a=20):
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
    a2[-1].s = flvl(2, 2.0)

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

    p1 = lpgz('../dat/p1.pgz')  # 2D dosages
    p5 = lpgz('../dat/p5.pgz')  # 2D dosages
    b1 = lpgz('../dat/b1.pgz')  # 3D indicators
    b5 = lpgz('../dat/b5.pgz')  # 3D indicators

    dm = p1.shape[-1] * np.power(2.0, [0, 1])
    dm = np.array(dm, 'i4')

    from rdm.sae import SAE
    a1 = SAE.from_dim(dm)
    t1 = Trainer(a1, p1, p1, err=CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    from copy import deepcopy
    a2 = deepcopy(a1)
    a2[+0] = Odr((dm[0], dm[1]), b1.shape[0], 1, w=a1[+0].w)
    a2[-1] = Odr((dm[1], dm[0]), b1.shape[0], 0, w=a1[-1].w)
    t2 = Trainer(a2, b1, b1, err=Odr.CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    return a1, a2, p1, p5, b1, b5, t1, t2


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
