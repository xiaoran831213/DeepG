# fine-tuner for neural networks
import numpy as np
import os
import sys
from os import path as pt
from trainer import Trainer as Tnr
from xutl import spz, lpz
sys.path.extend(['..'] if '..' not in sys.path else [])


def ftn_sae(nnt, __x, nft=None, ae0=None, ae1=None, **kwd):
    """ layer-wise unsupervised pre-training for
    stacked autoencoder.
    nnt: the stacked autoencoders
    __x: the inputs.

    nft: maximum number of epochs to go through for fine-tuning.
    ae0: which autoencoder in the stack to start the tuning?
    ae1: which autoencoder in the stack to end the tuning?

    By default, the the entire SAE of all layers are tuned, that is,
    start = 0, depth = len(nnt.sa)

    kwd: additional key words to pass on to the trainer.
    """
    # from hlp import dhom
    
    # build the trainer of fine-tuning:
    ftn = kwd.get('ftn', None)
    if ftn is None:
        n = nnt.sub(ae1, ae0) if ae0 or ae1 else nnt
        x = nnt.sub(ae0, 0).ec(__x).eval() if ae0 else __x
        ftn = Tnr(n, x, **kwd)

    # fine-tune
    nft = 20 if nft is None else nft
    ftn.tune(nft)

    kwd.update(ftn=ftn, nnt=nnt, __x=__x, nft=nft, ae0=ae0, ae1=ae1)
    kwd = dict((k, v) for k, v in kwd.iteritems() if v is not None)
    return kwd


def main(fnm='../../raw/W08/10_PTN', **kwd):
    """ the main fine-tune procedure, currently it only supports the Stacked
    Autoencoders(SAEs).

    fnm: pathname to the input, supposingly the saved progress after the pre-
    training. If {fnm} points to a directory, a file is randomly chosen from
    it.
    """
    # randomly pick pre-training progress if {fnm} is a directory and no record
    # exists in the saved progress:
    if isinstance(fnm, str):
        if pt.isdir(fnm):
            fnm = pt.join(fnm, np.random.choice(os.listdir(fnm)))
        kwd.update(fnm=fnm)
    else:
        fnm = kwd['fnm']

    fdt = lpz(fnm)
    fdt.update(kwd)
    kwd = fdt

    # check saved progress and overwrite options:
    sav = kwd.get('sav', pt.basename(fnm).split('.')[0])
    ovr = kwd.pop('ovr', 0)
    if sav and pt.exists(sav + '.pgz'):
        print(sav, ": exists,", )
        sdt = lpz(sav)
        if ovr is 0 or ovr > 2:  # do not overwrite the progress
            print(" skipped.")
            return sdt
        else:
            sdt.update(kwd)
            kwd = sdt

        # continue with the progress, but allow new or additional training data
        # (e.g. for validation).
        if ovr is 1:
            kwd['__x'] = fdt['__x'] if '__x' in fdt else kwd['__x']
            print("continue training.")

        # restart the fine-tuning, try to reuse saved training data.
        if ovr is 2:
            print("restart training.")
            del kwd['nnt']

    # do the fine-tuning:
    kwd = ftn_sae(**kwd)        # <-- __x, nnt, npt, ptn, ...

    # save the progress
    if sav:
        print("write to: ", sav)
        ftn = kwd.pop('ftn')
        spz(sav, kwd)
        kwd['ftn'] = ftn

    kwd = dict((k, v) for k, v in kwd.iteritems() if v is not None)
    return kwd
