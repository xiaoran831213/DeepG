import os
from os import listdir as ls
import os.path as pt
from os.path import join as pj

import numpy as np
from rdm.gdy import gdy_sae
from rdm.ftn import ftn_sae
from rdm.sae import SAE


def main(f='../raw/W08/00_GNO', out=None, **kwd):
    """ adaptive training for one autoendocer. """
    nnt = kwd.get('nnt')
    if nnt is None:
        dat = kwd.get('dat', np.load(pj(f, np.random.choice(ls(f)))))
        gmx, sbj = dat['gmx'], dat['sbj']

        # shuffle the observations
        idx = np.random.permutation(gmx.shape[0])
        gmx = gmx[idx, ]
        sbj = sbj[idx]

        # select untyped variants
        idx = np.random.binomial(1, .05, gmx.shape[-1])
        umx = gmx[:, :, idx == 1]
        gmx = gmx[:, :, idx == 0]

        # other genotype formats
        usg = umx.sum(-2, dtype='<i4')
        dsg = gmx.sum(-2, dtype='<i4')

    # training and validation data
    __x = kwd.get('__x', gmx.reshape(gmx.shape[0], -1).astype('f')[:1750])
    __u = kwd.get('__x', gmx.reshape(gmx.shape[0], -1).astype('f')[1750:])
    kwd.update(__x=__x, __u=__u)

    dim = __x.shape[-1]
    # dim = [dim] + [int(dim/2**_) for _ in range(-2, 5) if 2**_ <= dim]
    dim = [dim] + [30]

    # nnt[-1].shp = S(1.0, 'Shp', 'f')
    kwd.update(nnt=kwd.pop('nnt', SAE.from_dim(dim)))

    # pre-train and fine-tune
    # kwd.update(npt=10)
    # kwd = gdy_sae(**kwd)
    # kwd.update(npt=0, nft=2000)
    kwd.update(nft=5000)
    kwd = ftn_sae(**kwd)

    # request output:
    if out:
        import tempfile
        tpd = tempfile.mkdtemp()

        # genomic map
        np.savetxt(pj(tpd, 'gmp.txt'), dat['gmp'], '%d\t%d\t%s')

        # genomic data in dosage format
        np.savetxt(pj(tpd, 'dsg.txt'), dsg, '%d')

        # untyped variants in dosage format
        np.savetxt(pj(tpd, 'usg.txt'), usg, '%d')

        # high-order features
        np.savetxt(pj(tpd, 'hof.txt'), nnt.ec(__x).eval(), '%.8f')

        # meta information
        inf = open(pt.join(tpd, 'inf.txt'), 'w')
        for k, v in kwd['ftn'].__hist__[-1].items():
            inf.write('{}={}\n'.format(k, v))
        inf.close()

        # the neural network
        from xutl import spz
        spz(pj(tpd, 'nnt.pgz'), nnt)

        # pack the output, delete invididual files
        import tarfile
        import shutil
        tar = tarfile.open('{}.tgz'.format(out), 'w:gz')
        pwd = os.getcwd()
        os.chdir(tpd)

        [tar.add(_) for _ in os.listdir('.')]
        shutil.rmtree(tpd, True)

        os.chdir(pwd)
        tar.close()

    kwd.update(dat=dat)
    return kwd
