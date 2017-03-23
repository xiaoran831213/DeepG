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


def gdy_sae(w, x, u=None, nep=5, npt=5, **kwd):
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
        x_i = x
        u_i = x if u is None or u.size is 0 else u
        for i, a in enumerate(w.sa):
            # the trainer
            if ptn[i] is None:
                ptn[i] = Tnr(a, x=x_i, u=u_i, **kwd)
            else:
                ptn[i].x.set_value(x_i)
                ptn[i].u.set_value(u_i)

            ptn[i].tune(nep)

            # wire the data to the bottom of the tuple, the output
            # on top is the training material for next layer
            x_i, u_i = a.ec(x_i).eval(), a.ec(u_i).eval()

    kwd.update(nep=nep, npt=npt)
    return kwd


def main(fnm='../../raw/W09', **kwd):
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

    # dosage format and training format
    dsg = gmx.sum(-2, dtype='<i4')

    # mark some untyped subjects (hold out)
    usb = kwd.get('usb', 0.0)
    __i = np.random.choice(gmx.shape[+0], usb * gmx.shape[+0], False)
    usb = np.zeros(gmx.shape[+0], 'u1')
    usb[__i] = 1

    # mark some variants as untyped, if requested
    ugv = kwd.get('ugv', 0.0)
    __i = np.random.choice(gmx.shape[-1], ugv * gmx.shape[-1], False)
    ugv = np.zeros(gmx.shape[-1], 'u1')
    ugv[__i] = 1

    # collect untyped subjects and variants
    kwd['usb'] = usb        # untyped subject mask
    kwd['ugv'] = ugv        # untyped variant mask

    # training data
    xmx = gmx[usb == 0, :, :][:, :, ugv == 0]
    xmx = xmx.reshape(xmx.shape[0], -1)

    # options in {kwd} takes precedence over those loaded from {fnm}
    fdt.update(kwd, gmx=gmx, dsg=dsg, xmx=xmx)
    kwd = fdt

    # check saved progress and overwrite options:
    sav = kwd.get('sav', pt.basename(fnm).split('.')[0])
    ovr = kwd.pop('ovr', 0)
    if pt.exists(sav + '.pgz'):
        print(sav, ": exists,", )
        if ovr is 0 or ovr > 2:  # do not overwrite the progress
            print(" skipped.")
            return kwd

        # continue with the progress, new data in xmx can be used
        if ovr is 1:
            sdt = lpz(sav)
            kwd.pop('nwk', None)  # use saved network

            # options in keywords take precedence over saved ones
            sdt.update(kwd)
            kwd = sdt
            print("continue training.")

        # restart the progress, still uses new data xmx
        if ovr is 2:
            print("restart training.")
    kwd['sav'] = sav

    # create neural networks if necessary
    dim = xmx.shape[-1]
    dim = [dim] + [int(dim/2**_) for _ in range(1, 16) if 2**_ <= dim]
    if kwd.get('nwk') is None:
        kwd['nwk'] = SAE.from_dim(dim)

    # do the pre-training:
    kwd = gdy_sae(kwd['nwk'], xmx, xmx, **kwd)

    # save the progress
    if sav:
        print("write to: ", sav)
        spz(sav, kwd)
    return kwd
