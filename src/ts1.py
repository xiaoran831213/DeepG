import numpy as np

from rdm.trainer import Trainer
from rdm.pcp import Pcp
from rdm import hlp
from rdm.cat import Cat

from gsq.dsq import RndDsq
from copy import deepcopy
from pdb import set_trace
from pickle import load


def get_data(n=100, m=512, f='../raw/ann/03.vcf.gz'):
    """ get dosage data """
    # raw dosage value in {0, 1, 2}
    itr = RndDsq(f, wnd=m, dsg='011')
    dat = [[int(j) for j in next(itr)] for i in range(n)]

    # numpy float, rescaled to [0, 1]
    dat = np.array(dat, dtype='<f4')
    return dat


def get_SDA(dm=[1024, 512, 256, 128, 64, 32, 16, 8, 4]):
    """ create an stacked auto encoder """
    dms = list(zip(dm[:-1], dm[1:], range(len(dm))))
    ecs = [Pcp((i, j), tag='E{}'.format(t)) for i, j, t in dms]
    dcs = [Pcp((j, i), tag='D{}'.format(t)) for i, j, t in dms]

    # create audoendocers
    sda = list(zip(ecs, dcs))
    # constraint weight terms
    for ec, dc in sda:
        dc.w = ec.w.T
    return sda


def pre_train1(sda, trainers=None, rep=20):
    """ layer-wise pre-train."""

    # the trainers
    if trainers is None:
        tr = [None] * len(sda)
    else:
        tr = trainers

    # repeatitive pre-training
    for i in range(rep):
        dat = get_data(200, 512, '../raw/ann/03.vcf.gz')
        
        for i, l in enumerate(sda):
            # wire the encoder to its decoder counterpart
            ec, dc = l
            ae = Cat(ec, dc)

            # the trainer
            if tr[i] is None:
                tr[i] = Trainer(ae, src=dat, dst=dat, lrt=0.005)
            tr[i].tune(1)

            # wire the data to the bottom of the tuple, the output
            # on top is the training material for next layer
            dat = ec(dat).eval()

    return sda, tr


def fine_tune1(sda, trainer=None, dat=None):
    """ Fine tune the entire network."""
    # rewire encoders and decoders into a symmetric stack
    if type(sda[0]) is not Pcp:
        sda = deepcopy(sda)
        ecs, dcs = zip(*sda)
        nts = list(ecs) + list(reversed(dcs))
        sda = Cat(*nts)

    if dat is None:
        dat = get_data(200, 512, '../raw/ann/03.vcf.gz')
        
    if trainer is None:
        trainer = Trainer(sda, src=dat, dst=dat, lrt=0.0005)
    else:
        trainer.src.set_value(dat)
        trainer.dst.set_value(dat)
        
    trainer.tune(10)
    return sda, trainer


def test_two(nnt=None, dat=None):
    hlp.set_seed(None)

    if nnt is None:
        nnt = get_SDA([512, 1024, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4])

    nnt = pre_train1(nnt)
    trainer = Trainer(nnt, src=dat, dst=dat, lrt=0.005)
    trainer.tune()

    return nnt
