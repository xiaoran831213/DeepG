import numpy as np
from gsq.vsq import DsgVsq


def getd(m=256, f='../raw/wgs/03.vcf.gz', d='012'):
    """ get dosage data """
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


def rseq(wnd=256, dir='haf', sav=None, seed=None):
    """
    load sequence from a randomly drawn segment in the genome.
   
    wnd: the sampling windown size
    sav: where to save the samples
    """
    if seed:
        np.random.seed(seed)

    # draw genomic data
    chr = np.random.randint(22) + 1
    vcf = '{}/{:02d}.vcf.gz'.format(dir, chr)
    bp0 = np.random.randint(2 ^ 31)

    itr = DsgVsq(vcf, bp0=bp0, wnd=wnd, dsg=None)

    # genomic matrix
    gmx = next(itr)

    # fix MAF
    __i = np.where(gmx.sum((0, 1)) > gmx.shape[0])[0]
    gmx[:, :, __i] = 1 - gmx[:, :, __i]

    # fix copy (cpy0 >= cpy1).all() = True
    __i = np.where(gmx[:, 0, :] < gmx[:, 1, :])
    gmx[:, 0, :][__i] = 1
    gmx[:, 1, :][__i] = 0

    # subjects
    sbj = np.array(itr.sbj(), dtype='S16')

    # genomic map
    gmp = np.ndarray(
        wnd,
        dtype=np.dtype([('CHR', '<i1'), ('POS', '<i4'), ('UID', 'S32')]))
    gmp['CHR'] = itr.CHR
    gmp['POS'] = itr.pos()
    gmp['UID'] = itr.vid()

    # save and return
    if sav:
        np.savez_compressed(sav, gmx=gmx, gmp=gmp, sbj=sbj)
    return gmx, gmp, sbj
