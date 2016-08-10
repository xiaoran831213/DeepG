import numpy as np

import theano.tensor as T
from rdm.trainer import Trainer
from rdm.lyr import Lyr
from rdm import hlp

from gsq.dsq import RndDsq
from pdb import set_trace as ST


def get_data(n=100, m=512):
    """ get dosage data """
    # raw dosage value in {0, 1, 2}
    itr = RndDsq('../raw/ann/04.vcf.gz', wnd=m)
    dat = [[int(j) for j in next(itr)] for i in range(n)]

    # numpy float, rescaled to [0, 1]
    dat = np.array(dat, dtype='<f4')/2
    return dat


def get_SDA(dm, data=None):
    """ create an stacked auto encoder """
    rnd = np.random.RandomState(150)

    # figur out the inputing dimension
    if data is not None:
        dm = [data.shape[1]] + dm

    dms = list(zip(dm[:-1], dm[1:], range(len(dm))))
    ecs = [Lyr((i, j), tag='E{}'.format(t)) for i, j, t in dms]
    dcs = [Lyr((j, i), tag='D{}'.format(t)) for i, j, t in dms]

    # create audoendocers
    sda = list(zip(ecs, dcs))
    # constraint weight terms
    for ec, dc in sda:
        dc.w = ec.w.T
    return sda


def pre_train1(x, sda, nep=600):
    """ layer-wise pre-train."""
    for ec, dc in sda:
        # wire the encoder to its decoder counterpart
        ST()
        y_h = ec(x)
        x_p = dc(y_h)

        # the trainer
        tr = Trainer(x_p, src=x, dst=x, lrt=0.005)
        tr.tune(nep, npt=10)

        # wire the data to the bottom of the tuple, the output
        # on top is the training material for next layer
        ec.x(x)
        x = ec.y().eval()
    del x


def fine_tune1(x, sda, nep=600):
    """ Fine tune the entire network."""
    # re-wire encoders and decoders into a symmetric stack
    ecs, dcs = zip(*sda)
    sda = list(ecs) + list(reversed(dcs))
    for i, j in zip(sda[:-1], sda[1:]):
        j.x(i.y)  # lower output -> higher input

    tr = Trainer(sda[0].x, sda[-1].y, src=x, dst=x, lrt=0.0005)
    tr.tune(nep, npt=10)
    return tr


def test_one(x, out=None):
    """ test of one run. """

    import sys
    hlp.set_seed(1234)
    # decide the output
    if not out:
        fo = sys.stdout
    elif type(out) is str:
        fo = open(out, 'ab')
    else:
        fo = out

    # dimensionaly
    dim = [512, 256, 128, 64, 32, 16, 8]
    sda = get_SDA(dim, x)

    pre_train1(x, sda, nep=200)
    t = fine_tune1(x, sda, nep=200)

    line = '{}\t{}\t{}'.format(dim, t.cost(), t.gsum())
    fo.writelines(line)

    if type(out) is str:
        fo.close()

        return t
