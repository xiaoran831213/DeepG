# deep learning test
import numpy as np
from xnnt.tnr.bas import Base as Trainer
from xnnt.sae import SAE
import os
from os import path as pt
from xnnt.hlp import S


def main(fnm, nep=20, out=None, rdp=0, **kwd):
    """ Performance test for sigmoid, relu autoencoders.
    out: output location.
    rdp: reduce network depth.
    """

    # pick data file
    if(pt.isdir(fnm)):
        fnm = pt.join(fnm, np.random.choice(os.listdir(fnm)))
    dat = np.load(fnm)

    # convert to float 32 for GPU
    gmx = dat['gmx'].astype('f')
    
    # flatten the two copies of genotype: [copy1, copy2]
    gmx = gmx.reshape(gmx.shape[0], -1)
    N, D = gmx.shape[0], gmx.shape[1]

    # separate training and testing data
    i = np.zeros(N, 'i4')
    i[np.random.randint(0, N, N/5)] = 1
    x = gmx[i > 0]            # training data
    u = gmx[i < 1]            # testing data

    # train autoencoders
    # each layer half dimensionality
    d = [D]
    while d[-1] > 1:
        d.append(d[-1]/2)
    d[-1] = 1
    for i in range(rdp):
        d.pop()

    h = dict()
    # train the normal sigmoid network
    n = SAE.from_dim(d, s='sigmoid', **kwd)
    t = Trainer(n, x, u=u, lrt=1e-3, err='CE', **kwd)
    t.tune(nep)
    h['sigmoid'] = t.query()

    # relu + sigmoid
    n = SAE.from_dim(d, s='relu', **kwd)
    n[-1].s = 'sigmoid'
    t = Trainer(n, x, u=u, lrt=1e-3, err='CE', **kwd)
    t.tune(nep)
    h['relu_sigmoid'] = t.query()

    # relu + elliot, fixed shape
    n = SAE.from_dim(d, s='relu', **kwd)
    n[-1].s = 'elliot'
    t = Trainer(n, x, u=u, lrt=1e-3, err='CE', **kwd)
    t.tune(nep)
    h['relu_elliot_A1'] = t.query()

    # relu + elliot, flexible shape
    n = SAE.from_dim(d, s='relu', **kwd)
    n[-1].s = 'elliot'
    n[-1].alpha = S(1.0, 'f')
    t = Trainer(n, x, u=u, lrt=1e-3, err='CE', **kwd)
    t.tune(nep)
    h['relu_elliot_a1'] = t.query()
    h['alpha_1'] = n[-1].alpha.get_value()

    # save the training histories.
    if out is None:
        out = '.'
    if pt.isdir(out):
        out = pt.join(out, pt.basename(fnm).split('.')[0])
    np.savez_compressed(out, **h)

    return h
