# fine-tuner for neural networks
from functools import reduce
from numpy.random import permutation
from os import listdir as ls, path as pt

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bunch import Bunch
from scipy.stats import zscore
from sklearn.decomposition import PCA
from xnnt.mlp import MLP
from xnnt.tnr.basb import Base as Tnr
from xutl import spz, lpz

from gsm import sim


# r=main('../1kg/rnd/0000', seed=1, sav='../tmp', lr=1e-3, N=625, P=2000, dim=[4000, 50], frq=.5, fam='sin2.5', hvp=30, nep=500)
def main(fnm, **kwd):
    """ the fine-tune procedure for Stacked Autoencoder(SAE).
    -- fnm: filename to the input data.

    ********** keywords **********
    ** sav: where to save to simulation progress.

    **   N: number of samples to be used
    **   P: number of SNPs to be used

    ** nep: maximum number of epoches to go through for training
    ** lr: initial learning rate, will be overwriten by resumed progress

    ** xtp: type of x variable
    'gmx': genomic matrix       (N x P or N x 2P)
    'pcs': principle components (N x N)

    ** gtp: genomic data type
    'dsg': flattened genotype
    'xmx': dosage values

    ** dim: network dimensions for hidden units. the network dimension will
    be [2*P] + dim + [1] for flattened genotype input.

    ** seed: controls random number generation.

    ** svn: save network? it may take large amount of disk space and slow
    to reload.
    """
    from xutl import lpg, spg

    # for now, always start over the training.
    prg = kwd.pop('prg', 2)
    prg, kwd = lpg(fnm, prg=prg, vbs=1, **kwd)

    # should we even start?
    if prg == 0:
        print('NT: Done.')
        return kwd

    hte = kwd.pop('hte', 0.0)
    print('NT: HTE = {}'.format(hte))
    eot = kwd.get('eot', 1e9)
    print('NT: EOT = {}'.format(eot))
    if hte > eot:
        print('NT: eot < hte')
        print('NT: Halt.\nNT: Done.')
        return kwd

    seed = kwd.get('seed', None)
    if seed is not None:
        np.random.seed(seed)
        print('XT: SEED =', seed)

    # start over, simulate phenotype
    if prg == 2:
        gmx = kwd.pop('gmx').astype('f')
        maf = kwd.get('maf', None)
        if maf is not None:
            gmx = gmx[:, :, gmx.mean((0, 1)) >= maf]

        # pick genotype
        idx = permutation(gmx.shape[0])[:kwd.pop('N', 1000)]
        jdx = permutation(gmx.shape[2])[:kwd.pop('P', 2000)]
        gmx = gmx[idx, :, :][:, :, jdx]
        phe = sim(gmx, **kwd)  # puts in phenotype
        kwd.update(gmx=gmx, phe=phe)

    # -------------- y and x -------------- #
    gmx = kwd['gmx']
    phe = kwd['phe']
    dsg = gmx.sum(1)
    gtp = kwd.get('gtp', 'flt')
    if gtp == 'flt':  # flat format
        gmx = gmx.reshape(gmx.shape[0], -1)
    else:  # dosage format
        gmx = dsg

    phe = kwd['phe']
    xtp = kwd.get('xtp', 'gmx')
    if xtp == 'pcs':
        # perform PCA on genome data
        try:
            pca = PCA(n_components=min(gmx.shape[0], gmx.shape[1]))
            pcs = pca.fit_transform(gmx)
        except Exception as e:
            pcs = e
        xmx = pcs
    else:
        xmx = gmx

    # demanding z-scores?
    if kwd.get('zsc', 0):
        xmx = zscore(xmx, 0)

    # remove non-variables
    jdx = xmx.std(0) > 0
    if jdx.sum() < xmx.shape[1]:
        print('XT: exclude', xmx.shape[1] - jdx.sum(), 'null-info variants.')
        xmx = xmx[:, jdx]

    N = xmx.shape[0]
    div = int(kwd.get('div', 0.80) * N)
    xmk = zscore(dsg, 0) if kwd.get('zsc', False) else dsg
    xmk = np.concatenate([np.ones_like(phe), xmk], 1)

    if div < N:
        xT, xE = xmx[:div, :], xmx[div:, :]
        yT, yE = phe[:div, :], phe[div:, :]

        # for kernel methods
        xKT, xKE = xmk[:div, :], xmk[div:, :]
        yKT, yKE = yT, yE

        # further divide training set for NNT
        div = int(kwd.get('div', 0.80) * xT.shape[0])
        xT, xV = xT[:div, :], xT[div:, :]
        yT, yV = yT[:div, :], yT[div:, :]
    else:
        xT, xV, xE = xmx, None, None
        yT, yV, yE = phe, None, None

    # ---------- benchmark references ---------- #
    fam = kwd.get('fam', 'gau')
    if prg == 2:
        if fam == 'bin':
            from bmk import svclf, dtclf, nul
            bmk = pd.concat([
                svclf(xKT, yKT, xKE, yKE),  # Support vector
                dtclf(xKT, yKT, xKE, yKE)  # Decision Tree
            ])
        else:
            from bmk import knreg, svreg, dtreg, nul
            bmk = pd.concat([
                knreg(xKT, yKT, xKE, yKE),  # Kernel_ridge
                # svreg(xKT, yKT, xKE, yKE),  # Support vector
                # dtreg(xKT, yKT, xKE, yKE)   # Decision Tree
            ])
        bmk = bmk.append(nul(xKT, yKT, xKE, yKE, fam))

        # difference of error and relative error wrt. null model
        _lg = (bmk.mtd == 'nul', bmk.par == '-', bmk.key == 'ERR')
        bas = bmk.val[reduce(np.logical_and, _lg)][0]
        cpy = bmk[bmk.key == 'ERR'].copy()
        cpy.loc[:, 'val'] = cpy.val - bas
        cpy.loc[:, 'key'] = 'DFF'
        bmk = bmk.append(cpy)

        cpy = bmk[bmk.key == 'ERR'].copy()
        cpy.loc[:, 'val'] = cpy.val / bas
        cpy.loc[:, 'key'] = 'REL'
        bmk = bmk.append(cpy)

        kwd.update(bmk=pd.DataFrame(bmk))
    else:
        bmk = kwd['bmk']

    # train the network, create it if necessary
    nwk = kwd.pop('nwk', None)
    if nwk is None:
        # decide network shape
        dim = kwd.pop('dim', None)
        if dim is None:
            dim = min(*xmx.shape)
            dim = [dim, dim // 2, dim // 2]
        shp = [xmx.shape[1]] + dim + [1]
        sgm = 'sigmoid'
        nwk = MLP(shp, s=sgm, **kwd)
        if fam != 'bin':
            nwk[-1].s = 1
        kwd.update(dim=dim)
        print('create NT: ', nwk)

    # training
    kwd['err'] = 'CE' if fam == 'bin' else 'L2'
    kwd['hvp'] = kwd.get('hvp', np.inf)  # infinite patience
    kwd['bdr'] = 0  # no bold driving
    kwd['hte'] = hte
    kwd['bsz'] = 50
    if 'lr' not in kwd:
        kwd['lr'] = 1e-4
    ftn = Tnr(nwk, xT, yT, xV, yV, xg=xE, yg=yE, **kwd)
    ftn.tune(kwd.get('nep'))
    print('EOT=', ftn.terr())

    lr = ftn.lr.get_value()
    hst = ftn.hist()

    # record NNT performance on generalization set
    e = hst.loc[hst.verr.idxmin()]
    _lg = (bmk.mtd == 'nul', bmk.par == '-', bmk.key == 'ERR')
    bas = bmk.val[reduce(np.logical_and, _lg)][0]
    acc = [dict(key='ERR', val=e.gerr), dict(key='COR', val=e.gcor)]
    if fam == 'bin':
        acc.append(dict(key='AUC', val=e.gauc))
    acc.append(dict(key='DFF', val=e.gerr - bas))
    acc.append(dict(key='REL', val=e.gerr / bas))
    acc.append(dict(key='EPN', val=e.ep))
    acc = pd.DataFrame(acc)
    acc.loc[:, 'mtd'] = 'nnt'
    acc.loc[:, 'par'] = 'mve'
    bmk = bmk.append(acc)

    # report halting
    if ftn.hlt:
        print('NT: Halt=', ftn.hlt)

    # 3) update progress and save. for now, network is not saved to speed up
    # reporting, only the shape is saved.
    kwd.update(lr=lr, hst=hst, bmk=bmk)
    if (kwd.get('svn', False)):
        kwd.update(nwk=nwk)
    spg(**kwd)
    print('NT: Done.')

    kwd = dict((k, v) for k, v in kwd.items() if v is not None)
    kwd['nwk'] = nwk
    kwd['ftn'] = ftn
    return Bunch(kwd)


def collect(fdr='.', nrt=None, out=None, csv=None):
    """ collect simulation report in a folder. """
    fns = sorted(f for f in ls(fdr) if f.endswith('pgz'))

    # configuration, training history, and benchmarks
    cfg, hst, bmk = [], [], []
    for i, f in enumerate(fns):
        if nrt is not None and not i < nrt:
            break
        f = pt.join(fdr, f)
        print(f)
        pgz = lpz(f)

        # 1) collect training history
        hs = pgz.pop('hst')
        hst.append(hs)

        # 2) collect simulation configuration
        cf = ['fam', 'xtp', 'frq', 'mdl', 'rsq', 'gdy', 'gtp']
        cf = dict((k, v) for k, v in pgz.items() if k in cf)
        cf['nxp'] = '{}x{}'.format(pgz['gmx'].shape[0], pgz['gmx'].shape[2])
        cf['nwk'] = pgz['dim']
        cf = pd.Series(cf)
        cfg.append(cf)

        # 3) collect reference benchmarks, also append the performance of NNT
        bmk.append(pgz.pop('bmk').reset_index())

    # concatenation
    _df = []
    for c, b in zip(cfg, bmk):
        _df.append(pd.concat([pd.DataFrame([c] * b.shape[0]), b], 1))

    bmk = pd.concat(_df)
    # non-NNT methods do not rely on these parameters
    bmk.loc[bmk.mtd != 'nnt', ['gtp', 'nwk', 'xtp']] = '-'

    # configuration keys and report keys
    cfk = cf.index.tolist() + ['mtd', 'par', 'key']
    _gp = bmk.groupby(cfk)
    # means, stds, and iteration count of 'val'
    _mu = _gp.val.mean().rename('mu')
    _sd = _gp.val.std().rename('sd')
    _it = _gp.val.count().rename('itr')
    rpt = pd.concat([_mu, _sd, _it], 1).reset_index()
    rpt = rpt.loc[:, cfk + ['mu', 'sd', 'itr']]

    # do the same for training history
    hst = pd.concat(hst)
    _gp = hst.groupby('ep')
    _it = _gp.terr.count().rename('itr')
    hst = pd.concat([_gp.mean(numeric_only=True), _it], 1).reset_index()

    # save and return
    ret = Bunch(bmk=bmk, hst=hst, rpt=rpt)
    if out:
        spz(out, ret)
    if csv:
        rpt.to_csv(csv)
    return ret


def plot_hist(sim, out=None, gui=0):
    """ rearrange simulation pgz, report.
    sim: the outputs organized in a list of dictionaries.
    """
    if isinstance(sim, str):
        sim = lpz(sim)

    # performance measures
    import matplotlib as mpl
    if not gui:
        mpl.use('Agg')
    import matplotlib.pyplot as gc  # graphics context

    s0 = sim[0]
    x = s0['hst']['ep']  # shared x axis
    r2 = s0['rsq']  # shared r2
    frq = s0['frq']
    nep = s0['nep']
    fam = s0['fam']
    N, P = s0['gmx'].shape[0:3:2]

    # shared genome type
    gtp = s0['gtp']
    mdl = s0['mdl']
    ttl = dict(r2=r2, N=N, P=P, gtp=gtp, frq=frq, fam=fam, nep=nep, mdl=mdl)
    ttl = ','.join(['='.join([str(k), str(v)]) for k, v in ttl.items()])
    print(ttl)

    if out is None:
        out = '.'
    if pt.isdir(out):
        out = pt.join(out, ttl)

    # benchmarks
    bmk = np.concatenate([_['bmk'] for _ in sim])

    # course types
    xtp = set(_['xtp'] for _ in sim)
    cs = 'bgrcmykw'
    for i, t in enumerate(xtp):
        sub = [s for s in sim if s['xtp'] == t]
        nwk = str(sub[0]['dim'])

        # histories
        h = np.array([_['hst'] for _ in sub])

        # early stop
        early = np.argmin(h['verr'].mean(0))

        # error plot
        gc.subplot(2, 1, 1)
        gc.loglog(x, h['verr'].mean(0), c=cs[i], ls='-', lw=2, label=t)
        gc.loglog(x, h['terr'].mean(0), c=cs[i], ls='--', lw=2, label='_' + t)
        if i == len(xtp) - 1:
            gc.loglog([early, early], gc.ylim(), c=cs[i], ls='-', lw=2)

        if i == 0:
            gc.ylabel(r'error')
            gc.title(ttl)
        gc.legend()

        # correlation plot
        gc.subplot(2, 1, 2)
        acc = 'tauc' if fam == 'bin' else 'vcor'
        ylab = r'$auc(y, \hat{y})$' if fam == 'bin' else r'$corr(y, \hat{y})$'
        gc.loglog(x, h[acc].mean(0), c=cs[i], lw=2, label=t)
        if i == len(xtp) - 1:
            gc.loglog([early, early], gc.ylim(), c=cs[i], ls='-', lw=2)

        # record
        acc = h[acc].max(1).mean()
        pgz = np.array([('nnt', '{!s:>10}'.format(t), acc)], bmk.dtype)
        bmk = np.concatenate([bmk, pgz])
        print("DNN: {:s} {:s} {:.3f}".format(t, nwk, acc))

        # axis, labels should be plot only once
        if i == 0:
            # horizontal line to show r2
            gc.loglog(x, np.repeat(r2, x.size), 'r', lw=2, label=r'$r^2$')

            # other decoration elements
            gc.ylabel(ylab)
            gc.xlabel(r'epoch')
        gc.legend(loc=4)

    # fo = out + '.bmk'
    # np.savetxt(fo, bmk, '%s', header=' '.join(bmk.dtype.names), comments='')

    # fo = out + '.png'
    # gc.savefig(fo)

    return gc, bmk


# r1, p1 = plt1('~/1x1_gno.pgz')
def plt1(rpt, key='REL', log=True):
    """ plot supervised learning report. """
    # load report form file if necessary.
    sim = ['fam', 'frq', 'mdl', 'nxp']
    nnt = ['gtp', 'xtp', 'nwk']
    mtd = ['mtd', 'par']
    if isinstance(rpt, str) and rpt.endswith('pgz'):
        rpt = lpz(rpt)

    # the benchmark records
    bmk = rpt.bmk

    # title
    ttl = bmk.iloc[0][sim]
    ttl = ', '.join('{}={}'.format(k, v) for k, v in ttl.items())

    # method grouping
    grp = nnt + mtd

    # plot of relative error
    err = bmk[bmk.key == key].loc[:, nnt + mtd + ['val']]
    err = err[err.mtd != 'nul']

    # sample some data points to craft boxplot states
    X, L = [], []
    for l, g in err.groupby(grp):
        if 'nnt' in l:
            l = "{nwk:>10}.{mtd}".format(**g.iloc[0])
        else:
            l = "{par:>10}.{mtd}".format(**g.iloc[0])
        x = np.array(g.val)
        X.append(x)
        L.append(l)
    X = np.array(X).T
    S = cbook.boxplot_stats(X, labels=L)

    # plot
    plt.close('all')
    plt.title(ttl)
    ax = plt.axes()
    if log:
        ax.set_yscale('log')
    ax.bxp(S)

    # draw a line at y=1
    x0, x1 = ax.get_xbound()
    zx, zy = np.linspace(x0, x1, 10), np.ones(10)
    ax.plot(zx, zy, linestyle='--', color='red', linewidth=.5)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    return rpt, plt
