# greedy pre-trainer for neural networks
import numpy as np
import os
import sys
from os import path as pt
from xutl import spz, lpz
from tnr.cmb import Comb as Tnr
from sae import SAE
from hlp import cv_msk
sys.path.extend(['..'] if '..' not in sys.path else [])
from pdb import set_trace


def gdy_sae(w, x, u, nep=1, npt=20, **kwd):
    """ layer-wise unsupervised pre-training for
    stacked autoencoder.
    -- w: the network, i.e., the stacked autoencoder
    -- x: the training inputs.
    -- u: the validation data.
    
    -- nep: number of epoch to go through for each layer
    -- npt: number of pre-train to go through

    ** ptn: the pre-trainers for each sub AE of the SAE
    """
    # the trainers
    ptn = kwd.get('ptn', [None] * len(w.sa))

    # repeatitive pre-training
    for r in range(npt):
        x_i, u_i = x, x if u is None else x
        for i, a in enumerate(w.sa):
            # the trainer
            if ptn[i] is None:
                ptn[i] = Tnr(a, x=x_i, u=u_i, vdr=.03)
            if x_i is not None:
                ptn[i].x.set_value(x_i)
                ptn[i].u.set_value(u_i)

            ptn[i].tune(nep)

            # wire the data to the bottom of the tuple, the output
            # on top is the training material for next layer
            x_i, u_i = a.ec(x_i).eval(), a.ec(u_i).eval()

    kwd.update(ptn=ptn, npt=npt)
    return kwd


def main(fnm, **kwd):
    """ the main fine-tune procedure, currently it only supports the Stacked
    Autoencoders(SAEs).

    fnm: pathname to the input, supposingly the saved progress after the pre-
    training. If {fnm} points to a directory, a file is randomly chosen from
    it.
    """
    # randomly pick pre-training progress if {fnm} is a directory
    if pt.isdir(fnm):
        fnm = pt.join(fnm, np.random.choice(os.listdir(fnm)))
    kwd.update(fnm=fnm)

    # load data from {fnm}, also do some fixing:
    fdt = dict(np.load(fnm))
    gmx = fdt['gmx'].astype('f')  # fix data type
    __i = np.where(gmx.sum((0, 1)) > gmx.shape[0])[0]  # fix MAF
    gmx[:, :, __i] = 1 - gmx[:, :, __i]

    # dosage format and training format
    dsg = gmx.sum(-2, dtype='<i4')
    __x = gmx.reshape(gmx.shape[0], -1)
    fdt.update(gmx=gmx, dsg=dsg, __x=__x)

    # cross-validation
    cvk = kwd.get('cvk', 1)     # number of folds, default is 1 (no CV)
    cvm = cv_msk(__x, cvk)      # sample mask
    fdt.update(cvk=cvk, cvm=cvm)

    # options in {kwd} takes precedence over those loaded from {fnm}
    fdt.update(kwd)
    kwd = fdt

    # check saved progress and overwrite options:
    sav = kwd.get('sav', pt.basename(fnm).split('.')[0])
    ovr = kwd.pop('ovr', 0)
    if sav and pt.exists(sav + '.pgz'):
        print(sav, ": exists,", )
        if ovr is 0 or ovr > 2:  # do not overwrite the progress
            print(" skipped.")
            return kwd

        # parameters in {kwd} take precedence over those from {sav}.
        sdt = lpz(sav)
        sdt.update(kwd)
        kwd = sdt

        # continue with the progress
        if ovr is 1:
            print("continue training.")

        # restart the progress, reusing saved training data is OK.
        if ovr is 2:
            print("restart training.")
            del kwd['nnt']
    kwd['sav'] = sav

    # create neural network if necessary
    if kwd.get('nnt') is None:
        dim = __x.shape[-1]
        dim = [dim] + [int(dim/2**_) for _ in range(0, 16) if 2**_ <= dim]
        nws = [SAE.from_dim(dim) for _ in range(cvk)]
        kwd['nws'] = nws

    # do the pre-training:
    for i in range(cvk):
        kwd = gdy_sae(nws[i], __x[-cvm[i]], __x[+cvm[i]], **kwd)

    # save the progress
    if sav:
        print("write to: ", sav)
        ptn = kwd.pop('ptn')
        spz(sav, kwd)
        kwd['ptn'] = ptn
    return kwd
