import os.path as pt
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import pp
import hlp
from hlp import S
from hlp import F
import trainer
from trainer import Trainer
import lyr
from lyr import Lyr
import pdb

def get_data():
    x = np.load(pt.expandvars('$AZ_IMG1/rh02B21.npz'))['vtx']['tck']
    x = x.reshape(x.shape[0], -1)
    d = (x.shape[1], x.shape[1]/2)
    x = (x - x.min()) / (x.max() - x.min())
    return x

def get_das(dims, data = None):
    rnd = np.random.RandomState(150)
    if data is not None:
        dims = [data.shape[1]] + dims
    ZDT = zip(dims[:-1], dims[1:], xrange(len(dims)))
    ecs = [Lyr((i, j), np_rnd = rnd, tag = 'E{}'.format(t)) for i, j, t in ZDT]
    dcs = [Lyr((j, i), np_rnd = rnd, tag = 'D{}'.format(t)) for i, j, t in ZDT]

    das = zip(ecs, dcs)
    ## constraint weight terms
    for ec, dc in das:
        dc.w(ec.w, T.transpose)
    return das

def pre_train(data, das, nep = 600):
    x = data
    for ec, dc in das:
        dc.x(ec.y)
        tr = Trainer(ec.x, dc.y, src = x, dst = x, lrt = 0.005)
        tr.tune(nep, npt = 10)
        ec.x(x)
        x = ec.y().eval()
    del x

def fine_tune(data, das, nep = 600):
    x = data

    ## re-wire encoders and decoders
    ecs, dcs = zip(*das)
    sda = list(ecs) + list(reversed(dcs))
    for i, j in zip(sda[:-1], sda[1:]):
        j.x(i.y) # lower output -> higher input

    tr = Trainer(sda[0].x, sda[-1].y, src = data, dst = data, lrt = 0.0005)
    tr.tune(nep, npt= 10)
    return tr
    
def test_1(x, dims):
    fo = open('../../test_1.txt', 'ab')
    m = get_das(dims, x)
    pre_train(x, m, nep = 200)
    t = fine_tune(x, m, nep = 600)
    line = '{}\t{}\t{}\n'.format(dims, t.cost(), t.gsum())
    fo.write(line)
    fo.close()
    return t

def test_N(x):
    dims = [
        [512, 256, 128, 64, 32, 16, 8],
        [512, 256, 128, 64, 32, 16],
        [512, 256, 128, 64, 32],
        [512, 256, 128, 64],
        [512, 256, 128],
        [512, 256],
        [512],
        
        [256, 128, 64, 32, 16, 8],
        [256, 128, 64, 32, 16],
        [256, 128, 64, 32],
        [256, 128, 64],
        [256, 128],
        [256],
        
        [128, 64, 32, 16, 8],
        [128, 64, 32, 16],
        [128, 64],
        [128],

        [64, 32, 16, 8],
        [64, 32, 16],
        [64, 32],
        [64],

        [512, 512, 256, 256, 128, 128],
        [512, 512, 256, 256],
        [512, 512],

        [256, 256, 128, 128, 64, 64],
        [256, 256, 128, 128],
        [256, 256],

        [256, 256, 256, 256],
        [128, 128, 128, 128],
        [64, 64, 64, 64],
        [32, 32, 32, 32],
        [16, 16, 16, 16],
        [8, 8, 8, 8],
        [4, 4, 4, 4],

        [8, 8, 8],
        [4, 4, 4],
        [8, 8],
        [4, 4],
        [8],
        [4]
    ]

    for d in dims:
        test_1(x, d)

def test_Q(x):
    dims = [
        [1024, 512],
        [1024, 512, 256],
        [1024, 512, 256, 128]]

    for d in dims:
        test_1(x, d)
