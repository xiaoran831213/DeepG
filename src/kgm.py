# fine-tuner for neural networks
from os import path as pt, listdir as ls
import numpy as np
from xnnt.mlp import MLP
from xnnt.tnr.cmb import Comb as Tnr
from xutl import lpz, spz
from scipy.stats import zscore
from numpy.random import binomial, normal, permutation, choice
from sklearn.decomposition import PCA


def simu(gmx, **kwd):
    """ simulate phenotype frsom a segment of genome.
    ** frq: frequency of effective variable
    ** rsq: r-square for gaussian family, the deterministic coeficient.
    ** fam: distribution family
    """
    # handle genomic matrix
    gmx = lpz(gmx) if isinstance(gmx, str) else gmx
    gmx = gmx.sum(1)
    N, P = gmx.shape[0], gmx.shape[1]

    # addtive, dominent, and ressesive marks
    if kwd.get('mdl', 'a') == 'g':
        print('generate dominent and ressesive effect.')
        m = np.random.choice(3, P)
        gmx[:, m == 1] = (gmx[:, m == 1] > 0).astype('f') * 2  # dom
        gmx[:, m == 2] = (gmx[:, m == 2] > 1).astype('f') * 2  # res

    # weights drawn from Gaussian
    frq = kwd.get('frq', 0.30)
    w = binomial(1, frq, [P, 1]) * normal(size=[P, 1])

    # simulate signal
    xw = np.dot(gmx, w)           # linear signal

    # parity
    fpr = kwd.get('fpr', None)
    if fpr is not None:
        fpr = int(P * fpr) if fpr < 1 else fpr
        idx = gmx[:, choice(P, fpr)].sum(1) % 2
        msk = np.array([1, -1])[idx, np.newaxis]
        xw = xw * msk

    rsq = kwd.get('rsq', 0.30)
    if rsq > 0.0:
        rs = np.sqrt((1 - rsq) / rsq) * np.std(xw)
        sn = xw + np.random.normal(0.0, rs, [N, 1])  # signal + noise
    else:
        print('null effect simulated.')
        sn = np.random.normal(0.0, np.std(xw), [N, 1])

    fam = kwd.get('fam', 'gau')
    if fam == 'sin':
        phe = np.sin(sn)
    elif fam == 'bin':
        mu = rsq/(1 + np.exp(-xw)) + (1 - rsq) * 0.5
        phe = np.random.binomial(1, mu).reshape(N, 1)
    else:
        phe = sn
    phe = phe.astype('f')

    # update
    kwd.update(phe=phe, frq=frq, rsq=rsq, fam=fam)
    return kwd


def gemma(fnm, y, idx=None, jdx=None, div=.80):
    """ wrapper function for GEMMA """
    from tempfile import TemporaryDirectory
    from subprocess import run

    # create temporary directory
    tmd = TemporaryDirectory(prefix='', dir='.')
    dst = tmd.name
    gno = pt.join(dst, 'gno')

    # read *.fam file
    dtp = np.dtype([
        ('fid', '|U7'), ('iid', '|U7'),
        ('pid', '|U7'), ('mid', '|U7'),
        ('sex', '<i1'), ('phe', '<f4')])
    fam = np.genfromtxt('.'.join([fnm, 'fam']), dtp)

    # read *.bim file
    dtp = np.dtype([
        ('chr', '<i1'), ('snp', 'U32'),
        ('pos', '<i4'), ('gds', '<i4'),
        ('ref', 'U32'), ('alt', 'U32')])
    bim = np.genfromtxt('.'.join([fnm, 'bim']), dtp)

    # extract some subjects and snps
    idx = range(fam.size) if idx is None else idx
    jdx = range(bim.size) if jdx is None else jdx
    sbj, snp = pt.join(dst, 'sbj'), pt.join(dst, 'snp')
    fam, bim = fam[idx], bim[jdx]
    np.savetxt(sbj, fam[['fid', 'iid']], '%s', '\t')
    np.savetxt(snp, bim['snp'], '%s', '\t')
    cmd = 'plink --bfile {} --keep {} --extract {} --make-bed --out {}' \
          ' --memory 2048'.format(fnm, sbj, snp, gno).split()
    print(cmd)
    run(cmd)

    # relatedness matrix
    # cmd = 'gemma -bfile gno -gk 1'

    # fill phenotype into *.fam
    div = div * fam.size
    fam['phe'][:div] = y.flatten()[:div]
    np.savetxt('.'.join([gno, 'fam']), fam, '%s')

    # call gemma
    import os
    pwd = os.getcwd()
    os.chdir(dst)

    # 1) fitting
    cmd = 'gemma -bfile gno -maf 0 -bslmm 1'.format(gno, dst).split()
    print(cmd)
    run(cmd)
    # 2) prediction
    cmd = 'gemma -bfile gno -epm output/result.param.txt -emu' \
          ' output/result.log.txt -predict 1'.split()
    run(cmd)
    yht = np.genfromtxt('output/result.prdt.txt')[div:]
    os.chdir(pwd)

    # validation error, and correlation
    yts = y.flatten()[div:]
    err = ((yht - yts) ** 2).mean()
    cor = np.corrcoef(yts, yht)[0, 1]

    print('done, err=', err, 'cor=', cor)
    ret = dict(verr=err, vcor=cor, yht=yht, yts=yts)
    return ret


def kreg(xT, yT, xV=None, yV=None):
    """ Kernel Regression. """
    from sklearn.kernel_ridge import KernelRidge as KR
    if xV is None or yV is None:
        xV, yV = xT, yT

    # fitting
    kms = ['linear', 'poly', 'rbf', 'laplacian', 'sigmoid', 'cosine']
    ret = dict()
    for k in kms:
        clf = KR(alpha=1.0, kernel=k)
        clf.fit(xT, yT)

        # testing
        yH = clf.predict(xV)
        cR = np.corrcoef(yV.flatten(), yH.flatten())[0, 1]
        ret[k] = cR
    return ret


# r=main('../1kg/rnd/0004', sav='../tmp', N=500, P=1000, nep=200, xtp='pcs')
def main(fnm='../1kg/rnd/0001', **kwd):
    """ the fine-tune procedure for Stacked Autoencoder(SAE).
    -- fnm: pathname to the input, supposingly the saved progress after the
    pre-training. If {fnm} points to a directory, a file is randomly chosen
    from it.
    """
    if pt.isdir(fnm):
        fnm = pt.join(fnm, np.random.choice(ls(fnm)))

    fdr, fbn = pt.split(fnm)
    fpx = fbn.split('.', 2)[0]
    fnm = pt.join(fdr, fpx)
    kwd.update(fnm=fnm)

    # check saved progress and overwrite options:
    sav = kwd.get('sav', '.')
    if sav is None:
        sav = pt.join(fdr, fpx)
    if pt.isdir(sav):
        sav = pt.join(sav, fpx)
    if pt.exists(sav + '.pgz') or pt.exists(sav):
        print(sav, ": exists,", )
        ovr = kwd.pop('ovr', 2)  # overwrite?
        if ovr is 0 or ovr > 2:  # do not overwrite the progress
            print(" skipped.")
            return kwd
    else:
        ovr = 2

    # resume progress, use network stored in {sav}.
    if ovr is 1:
        # remaining options in {kwd} take precedence over {sav}, but always use
        # saved network, even if there is one available in {fnm}.
        kwd.pop('lrt', None)
        sdt = lpz(sav)
        sdt.update(kwd)
        kwd = sdt
        print("continue.")
    else:                       # restart the training
        print("restart.")

    # should we even start?
    hte = kwd.pop('hte', 0.0)
    eot = kwd.get('eot', 1e9)
    print('NT: HTE = {}'.format(hte))
    print('NT: EOT = {}'.format(eot))
    if hte > eot:
        print('NT: eot < hte')
        print('NT: Halt.\nNT: Done.')
        return kwd

    # read genomic matrix and subjects
    gmx = lpz(fnm + '.npz')['gmx']
    maf = kwd.get('maf', None)
    if maf is not None:
        gmx = gmx[:, :, gmx.mean((0, 1)) >= maf]

    # permute, then generate phenotype
    phe = kwd.get('phe', None)
    if phe is None:
        kwd['idx'] = permutation(gmx.shape[0])[:kwd.pop('N', 1000)]
        kwd['jdx'] = permutation(gmx.shape[2])[:kwd.pop('P', 2000)]
        gmx = gmx[kwd['idx'], :, :][:, :, kwd['jdx']]
        kwd = simu(gmx, **kwd)
        phe = kwd['phe']
    else:
        gmx = gmx[kwd['idx'], :, :][:, :, kwd['jdx']]

    # call gamma
    # gma = gemma(fnm, phe, kwd['idx'], kwd['jdx'])

    # flatten the genotype?
    gtp = kwd.get('gtp', 0)
    if gtp == 1:                # flat format
        gmx = gmx.reshape(gmx.shape[0], -1).astype('f')
    else:                       # dosage format
        gmx = gmx.sum(1)

    # perform PCA on genome data if necessary
    pcs = kwd.pop('pcs', None)
    if pcs is None:
        try:
            pca = PCA(n_components=min(gmx.shape[0], gmx.shape[1]))
            pcs = pca.fit_transform(gmx)
        except Exception as e:
            pcs = e
        kwd.update(pcs=pcs)

    # decide predictors
    xtp = kwd.get('xtp', 'gmx')
    if xtp == 'pcs':
        xmx = pcs
    else:
        xmx = gmx

    # demanding z-scores?
    if kwd.get('zsc', 0):
        xmx = zscore(xmx, 0)
    N = xmx.shape[0]
    xmx = xmx[:, xmx.std(0) > 0]
    dim = min(*xmx.shape)
    dim = [xmx.shape[1]] + [dim//2, dim//4, 1]

    div = int(kwd.get('div', 0.80) * N)
    if div < N:
        xT, xV = xmx[:div, :], xmx[div:, :]
        yT, yV = phe[:div, :], phe[div:, :]
    else:
        xT, xV = xmx, None
        yT, yV = phe, None

    # kernel_ridge regression
    krg = kreg(xT, yT, xV, yV)

    # learing rates
    lrt = kwd.pop('lrt', 1e-4)

    # train the network, create it if necessary
    nwk = kwd.pop('nwk', None)
    if nwk is None:
        nwk = MLP.from_dim(dim, **kwd)
        nwk[-1].s = 1
        print('create NT: ', nwk)

    # fine-tuning
    err = kwd.get('err', 'L2')
    if err is None:
        err = 'CE' if kwd['fam'] == 'bin' else 'L2'
        kwd['err'] = err

    nep = kwd.get('nep', 0)
    ftn = Tnr(nwk, xT, yT, xV, yV, lrt=lrt, bsz=50, err=err, hte=hte, **kwd)
    ftn.hvp = kwd.get('hpv', nep)
    ftn.bdr = kwd.get('bdr', 0)
    print('EOT=', ftn.terr())
    ftn.tune(nep)

    # ftn = MLP.Train(nwk, x=xT, y=yT, gmx=xV, pcs=yV, lrt=lrt, **kwd)
    lrt = ftn.lrt.get_value()
    yht = nwk(xmx).eval()
    hst = ftn.query()
    if ftn.hlt:
        print('NT: Halt,', ftn.hlt)

    # 3) update progress and save.
    kwd.update(nwk=nwk, lrt=lrt, phe=phe, yht=yht, hst=hst, krg=krg, hlt=ftn.hlt)

    if sav:
        print("write to: ", sav)
        spz(sav, kwd)
    print('NT: Done.')

    kwd = dict((k, pcs) for k, pcs in kwd.items() if pcs is not None)
    kwd['ftn'] = ftn
    return kwd


def collect(fdr):
    """ collect simulation output in a folder. """
    fns = sorted(f for f in ls(fdr) if f.endswith('pgz'))
    ret = []
    for i, f in enumerate(fns):
        # if not i < 300:
        #     break
        f = pt.join(fdr, f)
        print(f)
        output = lpz(f)
        dc = dict()
        dc['hst'] = output.pop('hst')[['ep', 'terr', 'verr', 'vcor']]
        dc.update(output)
        ret.append(dc)

    ret = [_ for _ in ret if _['hst'].shape[0] > _['nep']]
    return ret


def report(sim):
    """ rearrange simulation output, report.
    sim: the outputs organized in a list of dictionaries.
    """
    if isinstance(sim, str):
        sim = lpz(sim)

    # performance measures
    from matplotlib import pyplot as gc  # graphics context

    x = sim[0]['hst']['ep']             # shared x axis
    r2 = sim[0]['rsq']                  # shared r2
    gv = dict()
    for k in sim[0]['krg'].keys():
        gv[k] = np.array([_['krg'][k] for _ in sim]).mean()

    # course types
    types = set(_['xtp'] for _ in sim)
    cs = 'bgrcmykw'
    for i, t in enumerate(types):
        h = np.array([s['hst'] for s in sim if s['xtp'] == t])

        # early stop
        early = np.argmin(h['verr'].mean(0))

        # error plot
        gc.subplot(2, 1, 1)
        gc.loglog(x, h['verr'].mean(0), c=cs[i], ls='-',  lw=2, label=t)
        gc.loglog(x, h['terr'].mean(0), c=cs[i], ls='--', lw=2, label='_'+t)
        gc.loglog([early, early], gc.ylim(), c=cs[i], ls='-', lw=2)

        if i == 0:
            gc.ylabel(r'error')
        gc.legend()

        # correlation plot
        gc.subplot(2, 1, 2)
        gc.loglog(x, h['vcor'].mean(0), c=cs[i], lw=2, label=t)
        gc.loglog([early, early], gc.ylim(), c=cs[i], ls='-', lw=2)
        print(gv)
        print(h['vcor'].mean(0).max(), h['vcor'].max(1).mean())

        if i == 0:
            gc.loglog(x, np.repeat(r2, x.size), 'r', lw=2, label=r'$r^2$')
            # gc.loglog(x, np.repeat(gv, x.size), 'k', lw=2, label=str(gv)[:3])
            gc.ylabel(r'$corr(y, \hat{y})$')
            gc.xlabel(r'epoch')
        gc.legend(loc=4)
    return gc
