import numpy as np
from utl import spgz


def getd(m=256, f='../raw/wgs/03.vcf.gz', d='012'):
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


def gend():
    d, p, b = getd(m=256)
    d1, d5 = d[..., :1750, :], d[..., 1750:, :]
    p1, p5 = p[..., :1750, :], p[..., 1750:, :]
    b1, b5 = b[..., :1750, :], b[..., 1750:, :]
    spgz('../dat/d1.pgz', d1)
    spgz('../dat/d5.pgz', d5)
    spgz('../dat/p1.pgz', p1)
    spgz('../dat/p5.pgz', p5)
    spgz('../dat/b1.pgz', b1)
    spgz('../dat/b5.pgz', b5)
