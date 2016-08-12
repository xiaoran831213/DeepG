import numpy as np

from rdm.trainer import Trainer
from rdm.pcp import Pcp
from rdm import hlp
from rdm.cat import Cat

from gsq.dsq import RndDsq
from copy import deepcopy
from pdb import set_trace


def get_data(n=100, m=512, f='../raw/ann/03.vcf.gz'):
    """ get dosage data """
    # raw dosage value in {0, 1, 2}
    itr = RndDsq(f, wnd=m, dsg='011')
    dat = [[int(j) for j in next(itr)] for i in range(n)]

    # numpy float, rescaled to [0, 1]
    dat = np.array(dat, dtype='<f4')
    return dat


def get_SDA(dm):
    """ create an stacked auto encoder """
    dms = list(zip(dm[:-1], dm[1:], range(len(dm))))
    ecs = [Pcp((i, j), tag='E{}'.format(t)) for i, j, t in dms]
    dcs = [Pcp((j, i), tag='D{}'.format(t)) for i, j, t in dms]

    # constraint weight terms
    for ec, dc in zip(ecs, dcs):
        dc.w = ec.w.T

    # wire them up
    dcs.reverse()
    stk = Cat(*ecs, *dcs)

    return stk


def pre_train1(stk, trainers=None, rep=1):
    """ layer-wise pre-train."""
    # the depth
    dp = len(stk)//2

    # the trainers
    if trainers is None:
        tr = [None] * dp
    else:
        tr = trainers

    m = stk[0].dim[0]
    # repeatitive pre-training
    for i in range(rep):
        dat = get_data(400, m, '../raw/ann/03.vcf.gz')

        for i in range(dp):
            # wire the encoder to its decoder counterpart
            ec, dc = stk[i], stk[-1-i]
            ae = Cat(ec, dc)

            # the trainer
            if tr[i] is None:
                tr[i] = Trainer(ae, src=dat, dst=dat, lrt=0.005, bsz=1)
            else:
                tr[i].src.set_value(dat)
                tr[i].dst.set_value(dat)
            tr[i].tune(1)

            # wire the data to the bottom of the tuple, the output
            # on top is the training material for next layer
            dat = ec(dat).eval()

    return stk, tr


def fine_tune1(stk, trainer=None, rep=1):
    """ Fine tune the entire network."""
    m = stk[0].dim[0]

    for i in range(rep):
        dat = get_data(400, m, '../raw/ann/03.vcf.gz')

        if trainer is None:
            trainer = Trainer(stk, src=dat, dst=dat, lrt=0.0005)
        else:
            trainer.src.set_value(dat)
            trainer.dst.set_value(dat)
        trainer.tune(1)
    
    return stk, trainer


def test_two(nnt=None, dat=None):
    hlp.set_seed(None)

    if nnt is None:
        nnt = get_SDA([512, 512, 256, 256, 128, 128, 64, 64])

    nnt, tr1 = pre_train1(nnt, rep=2)
    print()
    nnt, tr2 = fine_tune1(nnt, rep=2)

    return nnt


def AUC(x, z):
    """ Calculate Area Under the Curve."""
    from sklearn.metrics import roc_auc_score
    x = x.reshape(x.shape[0], -1)
    z = z.reshape(z.shape[0], -1)

    s = np.array([roc_auc_score(x[i], z[i]) for i in range(x.shape[0])])
    return s.mean()


def spgz(fo, s):
    """ save python object to gziped pickle """
    import gzip
    import pickle
    with gzip.open(fo, 'wb') as gz:
        pickle.dump(s, gz, -1)


def lpgz(fi):
    """ load python object from gziped pickle """
    import gzip
    import pickle
    with gzip.open(fi, 'rb') as gz:
        return pickle.load(gz)
