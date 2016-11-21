# k fold cross validation for NNT trainer
import numpy as np
from collections import OrderedDict as OD
from itertools import product
from copy import deepcopy as DC
try:
    from trainer import Trainer as Tnr
    from sae import SAE
    import hlp
except ValueError:
    from .trainer import Trainer as Tnr
    from .sae import SAE
    from . import hlp
import sys
import os
from xutl import spz, lpz

from pdb import set_trace


# the default lambda grid, with Ne-M format
__lmd__ = [
    (0, 0), (1, -4), (1, -3), (5, -3), (1, -2), (2, -2), (5, -2), (1, -1)]


def cv_msk(x, k, permu=True):
    """ return masks that randomly partition the data into {k} parts.
    x: the sample size or the sample data
    k: the number of partitions.
    """
    n = x if isinstance(x, int) else len(x)
    idx = np.array(np.arange(np.ceil(n / float(k)) * k), '<i4')
    if permu:
        idx = np.random.permutation(idx)
    idx = idx.reshape(k, -1) % n
    msk = np.zeros((k, n), 'bool')
    for _ in range(k):
        msk[_, idx[_, ]] = True
    return msk


# cross validation for stacked auto-encoders
def cv_sae_lmd_one(nnt, __x, k=4, **kwd):
    """ One round of cross-validation for lambda the regulator
    coeficient.

    nnt: the neural network (stacked autoencoder) to be trained
    __x: inputs for semi-unsupervised training

    k  : the number of CV folds (k-folds CV)

    ftn: the fine-tune trainer
    nft: number of fine tune to go through
    """
    # divide the observation by indices
    msk = kwd.get('msk', cv_msk(__x, k))

    # grid of lambda(s)
    lmd = kwd.get('lmd', __lmd__[:])

    # copies of network for each lambda times each CV fold
    nts = kwd.get('nts', OD((l, [DC(nnt) for _ in range(k)]) for l in lmd))

    # fine-tune trainer and epoch count
    print('compile fine tuner: ', 'ftn' not in kwd)
    ftn = kwd.get('ftn', Tnr(DC(nnt), x=__x, u=__x, lrt=.001))
    print('done compiling.')
    
    ftn.__hist__ = []
    nft = kwd.get('nft', 10)

    # history
    hst = kwd.get('hst', OD((l, [[] for _ in range(k)]) for l in lmd))
    print("begin CV.")
    sys.stdout.flush()
    for l, j in product(nts.keys(), range(k)):
        # the j.th network for lamda=l, j = 0 ..., k
        ftn.x.set_value(__x[+msk[j], ])      # for training
        ftn.u.set_value(__x[-msk[j], ])      # for validation
        ftn.lmd.set_value(l[0] * 10 ** l[1])  # set lambda

        # paste parameters to trainer's network
        hlp.dpcp(nts[l][j], ftn.nnt)
        ftn.__hist__ = []
        if len(hst[l][j]) > 0:
            ftn.ep.set_value(hst[l][j][-1]['ep'])
        else:
            ftn.ep.set_value(0)

        # tuning, and reporting
        print(l, j, id(nts[l][j]))
        ftn.tune(nft)

        # accumulate trainer history
        hst[l][j].extend(ftn.__hist__)

        # paste parameters back to external network
        hlp.dpcp(ftn.nnt, nts[l][j])

    kwd.update(
        nnt=nnt, __x=__x, k=k, msk=msk, hst=hst, nts=nts, ftn=ftn, nft=nft)
    return kwd


def cv_sae_lmd_all(lmt=20, lbk=20, **p):
    """ overall cross-validation."""

    for i in range(lmt):
        # break the CV if only one lambda was left.
        if len(p.get('hst', 'nothing')) < 2:
            print('only one lambda left, done CV.')
            break

        # one round of k-fold CV
        p = cv_sae_lmd_one(**p)

        # decide number of epochs to look back
        m = min(lbk, p['nft'], min(len(h) for h in p['hst'].viewvalues()))

        # only retain lambdas that reduces validation error
        hst, nts = OD(), OD()
        for l, h in p['hst'].items():
            e1 = sum(_[-1]['verr'] for _ in h)  # most recent eov
            e0 = sum(_[-m]['verr'] for _ in h)  # looked back eov
            print(l, e0, e1, e0 - e1)

            # drop the network for lambda=l if the latest validation error is
            # greater than the looked back value.
            if e0 < e1:
                continue

            # otherwise, retain them
            hst[l] = h
            nts[l] = p['nts'][l]

        p['hst'], p['nts'] = hst, nts

    return p


# depth of SAE
def cv_sae_lyr_one(nnt, __x, k=4, **kwd):
    pass


# for testing purpose
def rdat(fdr='../../raw/H08_20', seed=None):
    # pick data file
    np.random.seed(seed)
    fnm = np.random.choice(os.listdir(fdr))
    dat = np.load(os.path.join(fdr, fnm))
    gmx = dat['gmx'].astype('f')

    # fix MAF > .5
    __i = np.where(gmx.sum((0, 1)) > gmx.shape[0])[0]
    gmx[:, :, __i] = 1 - gmx[:, :, __i]
    __x = gmx.reshape(gmx.shape[0], -1)

    # set up neral network
    # from exb import Ods
    dim = __x.shape[-1]
    
    # dm8 = [dim, dim * 8, dim * 4]
    # dm4 = [dim, dim * 4, dim * 2]
    dm2 = [dim, dim * 2]
    # dm1 = [dim, dim, dim]
    # nt8 = SAE.from_dim(dm8)
    # nt4 = SAE.from_dim(dm4)
    nt2 = SAE.from_dim(dm2)
    # nt1 = SAE.from_dim(dm1)
    # nnt[-1].shp = S(1.0, 'Shp')

    # dat = {'__x': __x, 'nt8': nt8, 'nt4': nt4, 'nt2': nt2}
    return {'__x': __x, 'nnt': nt2}


def main():
    d = rdat()
    x, n = d['__x'], d['nnt']
    t1 = Tnr(hlp.paint(n), x, inc=1.04, dec=0.85)
    t2 = Tnr(hlp.paint(n), x, inc=1.00, dec=1.00)
    t2.__onep__ = None
    return t1, t2


def tst2():
    """ train the second layer of SAE."""
    x = lpz('../../dat/h0.pz')
    n = SAE.from_dim([x.shape[-1], x.shape[-1]/2])
    t1 = Tnr(hlp.paint(n), x, inc=1.04, dec=0.85)
    t2 = Tnr(hlp.paint(n), x, inc=1.00, dec=1.00)
    t2.__onep__ = None
    return t1, t2
    
