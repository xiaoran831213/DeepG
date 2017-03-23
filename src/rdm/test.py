# deep learning test
import numpy as np
import sys
from tnr.cmb import Comb as Tnr
from sae import SAE
import os
from theano import tensor as T
from hlp import S
from pdb import set_trace

sys.path.extend(['..'] if '..' not in sys.path else [])


# for testing purpose
def rdat(fdr='../../raw/W09/0003.npz'):
    # pick data file
    if(os.path.isdir(fdr)):
        fnm = np.random.choice(os.listdir(fdr))
        fnm = os.path.join(fdr, fnm)
    else:
        fnm = fdr
    
    dat = np.load(fnm)

    # categorial format
    c1 = dat['gmx'][:, np.newaxis, 0, :]
    c2 = dat['gmx'][:, np.newaxis, 1, :]
    c0 = 1 - (c1 + c2 > 0)
    cmx = np.concatenate((c0, c1, c2), 1).astype('f')

    # float 32 for GPU computation
    gmx = dat['gmx'].astype('f')
    gmx = gmx.reshape(gmx.shape[0], -1)

    __x = gmx[:1750]
    __u = gmx[1750:]
    _cx = cmx[:1750]
    _cu = cmx[1750:]

    return {'__x': __x, '__u': __u, '_cx': _cx, '_cu': _cu}


def main():
    d = rdat()
    x, u = d['__x'], d['__u']
    # cx, cu = d['_cx'], d['_cu']
    d = x.shape[-1]
    d = [d] + [d/2] + [d/4] + [d/8] + [d/16] + [d/32]
   
    n1 = SAE.from_dim(d, s='sigmoid')
    n1[-1].s = 'sigmoid'
    
    n2 = SAE.from_dim(d, s='sigmoid')
    n2[-1].s = 'sigmoid'
    
    # n3 = SAE.from_dim(d, s='relu')
    # n3[-1].s = 'sigmoid'

    # from pcp import Pcp
    # n2[-1] = Pcp(dim=[n2[-1].dim[0], cx.shape[-1]], s='softmax', cat=3)
    # n3[-1] = Pcp(dim=[n3[-1].dim[0], n3[-1].dim[1]], s='sigmoid')

    t1 = Tnr(n1, x, u=u, lrt=1e-2, err='CE', lmd=1e-4, bdr=0)
    t2 = Tnr(n2, x, u=u, lrt=1e-2, err='CE', lmd=1e-4, bdr=0)

    # z = np.zeros((x.shape[0], 2, x.shape[-1]), 'i1')
    # z[:, 0, :] = x == 0
    # z[:, 1, :] = x == 1
    return t1, t2
