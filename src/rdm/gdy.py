# greedy pre-trainer for neural networks
import numpy as np
import os
import sys
from os import path as pt
import sys
from xutl import spz, lpz
from trainer import Trainer as Tnr
from sae import SAE
sys.path.extend(['..'] if '..' not in sys.path else [])


def gdy_sae(nnt, __x, nep=1, npt=20, **kwd):
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
                ptn[i] = Tnr(a, x=x_i, u=x_i)
            if x_i is not None:
                ptn[i].x.set_value(x_i)
                ptn[i].lrt_inc.set_value(1.02)

            ptn[i].tune(nep)

            # wire the data to the bottom of the tuple, the output
            # on top is the training material for next layer
            x_i = a.ec(x_i).eval()

    kwd.update(ptn=ptn, nnt=nnt, __x=__x, npt=npt)
    return kwd


def main(fnm, **kwd):
    # check previously saved progress
    pz = kwd.get('pz', None)
    if pz:
        kwd.update(lpz(pz))

    # overwrite check
    sav, ovr = kwd.get('sav'), kwd.get('ovr')
    if sav and pt.exists(sav) and not ovr:
        print(sav, "exists: ", sav)
        return kwd

    # load training data
    if kwd.get('__x') is None:
        gmx = np.load(fnm)['gmx'].astype('f')
        # fix MAF > .5
        __i = np.where(gmx.sum((0, 1)) > gmx.shape[0])[0]
        gmx[:, :, __i] = 1 - gmx[:, :, __i]

        # dosage format and training format
        dsg = gmx.sum(-2, dtype='<i4')
        __x = gmx.reshape(gmx.shape[0], -1)
        kwd.update(gmx=gmx, dsg=dsg, __x=__x)

    # load neural network
    if kwd.get('nnt') is None:
        dim = __x.shape[-1]
        dim = [dim] + [int(dim/2**_) for _ in range(-2, 16) if 2**_ <= dim]
        nnt = SAE.from_dim(dim)
        kwd.update(nnt=nnt)
        
    # do the pre-training
    kwd = gdy_sae(**kwd)        # <-- __x, nnt, npt, ptn, ...

    # save the progress
    if sav:
        print("write to: ", sav)
        ptn = kwd.pop('ptn')
        spz(sav, kwd)
        kwd['ptn'] = ptn
    return kwd


# for testing purpose
def rdat(fdr='../../raw/W08/00_GNO', seed=None):
    # pick data file
    np.random.seed(seed)
    fnm = np.random.choice(os.listdir(fdr))
    fdr = pt.abso
    fnm = pt.join(fdr, fnm)
    return fnm
