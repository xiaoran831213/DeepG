# greedy pre-trainer for neural networks
import numpy as np
import os
from os import path as pt
try:
    from trainer import Trainer
    from hlp import S
    from sae import SAE
except ValueError:
    from .trainer import Trainer
    from .hlp import S
    from .sae import SAE
from xutl import spz, lpz


def gdy_sae(nnt, __x, nep=20, npt=1, **kwd):
    """ layer-wise unsupervised pre-training for
    stacked autoencoder.
    nnt: the stacked autoencoders
    __x: the inputs.
    ave: the filename to save the progress.

    ptn: the pre-trainers for each sub AE of the SAE
    npt: number pf greedy pre-train to go through

    kwd: key words to pass on to the trainer.
    -- ptn: the pre-trainers
    """
    # the trainers
    ptn = kwd.get('ptn', [None] * len(nnt.sa))

    # repeatitive pre-training
    for r in range(npt):
        x_i = __x
        for i, a in enumerate(nnt.sa):
            # the trainer
            if ptn[i] is None:
                ptn[i] = Trainer(a, x=x_i, u=x_i)
            if x_i is not None:
                ptn[i].x.set_value(x_i)

            ptn[i].tune(1)

            # wire the data to the bottom of the tuple, the output
            # on top is the training material for next layer
            x_i = a.ec(x_i).eval()

    kwd.update(nnt=nnt, __x=__x, ptn=ptn, npt=npt)
    return kwd


def main(**kwd):
    # check filename
    pz = kwd.get('pz', None)
    if pz:
        pz = lpz(pz)
        pz.update(**kwd)
        kwd = pz

    sav = kwd.get('sav')
    ovr = kwd.get('ovr', False)
    if sav and pt.exists(sav):
        print "exists: ", sav
        if not ovr:
            print ", skip."
            return kwd

    # save
    if sav:
        if pt.exists(sav):
            print "overwrite: ", sav
        else:
            print "write: ", sav
        spz(sav, kwd)
    return kwd


# for testing purpose
def rdat(fdr='../../raw/H08', seed=None):
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
    dim = [dim] + [int(dim/2**_) for _ in range(-2, 32) if 2**_ <= dim]
    nnt = SAE.from_dim(dim)
    nnt[-1].shp = S(1.0, 'Shp')

    dat = {'__x': __x, 'nnt': nnt, 'gmx': gmx}
    return dat
