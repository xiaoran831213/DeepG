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
from utl import hist


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


def ts5():
    """ adaptive training for one autoendocer. """
    hlp.set_seed(None)

    # {0, .5, 1}
    p1 = lpgz('../dat/p1.pgz')
    p5 = lpgz('../dat/p5.pgz')

    dm = p1.shape[-1] * np.power(2.0, [0, 1, 0, -1])
    dm = np.array(dm, 'i4')

    from rdm.sae import SAE
    from copy import deepcopy

    a1 = SAE.from_dim(dm)
    t1 = Trainer(a1, p1, p1, err=CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    a2 = deepcopy(a1)
    alpha2 = hlp.S(.2)
    a2[-1].s = flvl(2, alpha2)
    t2 = Trainer(a2, p1, p1, err=CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    # a3 = deepcopy(a1)
    # alpha3 = hlp.S(np.ones((1, p1.shape[-1])))/3
    # a3[-1].s = flvl(2, alpha3)
    # t3 = Trainer(a3, p1, p1, err=CE, reg=R1, lmd=.0, lrt=0.005, bsz=10)

    return a1, a2, p1, p5, t1, t2
