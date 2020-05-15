# fine-tuner for neural networks
from functools import reduce
from numpy.random import permutation
import numpy as np
import pandas as pd
from bunch import Bunch
from scipy.stats import zscore
from sklearn.decomposition import PCA

from gsm import sim


# r=main('../1kg/0000.npz', seed=1, sav='../tmp', N=620, P=1000, frq=.5, fam='g')
# r=main('../1kg/0000.npz', seed=1, sav='../tmp', N=625, P=1000, frq=.5, fam='sin.8')
def main(fnm, **kwd):
    """ the fine-tune procedure for Stacked Autoencoder(SAE).
    -- fnm: filename to the input data.

    ********** keywords **********
    ** sav: where to save to simulation progress.

    **   N: number of samples to be used
    **   P: number of SNPs to be used

    ** xtp: type of x variable
    'gmx': genomic matrix       (N x P or N x 2P)
    'pcs': principle components (N x N)

    ** gtp: genomic data type
    'dsg': flattened genotype
    'xmx': dosage values

    ** mdl: the model that composes genomic signals from the genotype.
    a  : the signal is the weighted sum of variants
    g  : the signal is the weighted sum of additive, dominance, and recessive
    variants

    g*g: the same as g, with additional 2nd order and crossover terms,
         e.g., g.j + g.k + g.j^2 + g.k^2 + g.j:g.k, for j, k = (1, .. P);

    a*a: the same as a, with additional 2nd order and crossover terms,
         e.g., a.j + a.k + a.j^2 + a.k^2 + a.j:a.k, for j, k = (1, .. P);

    g:g: the same as g*g but excluding 1st and 2nd order terms (e.g. g.i^2
    and g.i);
    a:a: the same as a*a but excluding 1st and 2nd order terms (e.g, a.i^2
    and a.i);

    ** fam: the family of the function that converts the genomic signal to
    output labels directly, or, to the parameter of some distribution that
    generate the labels.
    gau: gaussian family, which means no conversion, that is, y = s;
    bin: binomial family, the inverse logit conversion, that, is
    Pr(y==1) = 1/(1 + exp(-s)). y has to be separately drawn from the
    probabilities converted from s;
    sin: y = sin(s/sd(s) * period);

    ** seed: controls random number generation.

    """
    seed = kwd.get('seed') or None
    if seed is not None:
        np.random.seed(seed)
        print('XT: SEED =', seed)

    # get genotype matrix: 2504 samples, 2 copies, 2^14 variants
    gmx = np.load(fnm)['gmx'].astype('f')

    # restrict minimum minor allele frequency
    min_maf = kwd.get('maf', 0.0)
    gmx = gmx[:, :, gmx.mean((0, 1)) >= min_maf]

    # pick N genotype samples and P variants (i.e., SNPs)
    idx = permutation(gmx.shape[0])[:kwd.pop('N', 1000)]
    jdx = permutation(gmx.shape[2])[:kwd.pop('P', 2000)]
    gmx = gmx[idx, :, :][:, :, jdx]

    # simulate phenotype
    phe = sim(gmx, **kwd)

    # put genomic matrix and phenotype in the dictionary
    kwd.update(gmx=gmx, phe=phe)

    # -------------- y and x -------------- #
    # convert 2-copy format into allele dosage format
    dsg = gmx.sum(1)

    # which genotype format to use for our method?
    gtp = kwd.get('gtp', 'flt')
    if gtp == 'flt':            # 1) flat format
        gmx = gmx.reshape(gmx.shape[0], -1)
    else:                       # 2) dosage format
        gmx = dsg

    # which type of X to use for our method?
    xtp = kwd.get('xtp', 'gmx')
    if xtp == 'pcs':            # principle components
        try:
            # perform PCA on genome data
            pca = PCA(n_components=min(gmx.shape[0], gmx.shape[1]))
            pcs = pca.fit_transform(gmx)
        except Exception as e:
            pcs = e
        xmx = pcs
    else:
        xmx = gmx

    # demanding z-score standardization on X?
    if kwd.get('zsc', 0):
        xmx = zscore(xmx, 0)

    # remove non-features
    jdx = xmx.std(0) > 0
    if jdx.sum() < xmx.shape[1]:
        print('XT: exclude', xmx.shape[1] - jdx.sum(), 'null-info features.')
        xmx = xmx[:, jdx]

    # divide Training, Validation, and Evaluation datasets.
    N = xmx.shape[0]
    div = int(kwd.get('div', 0.80) * N)

    # input data prepared for competitors: kernel regression
    # kernel methods always use dosage format
    xmk = zscore(dsg, 0) if kwd.get('zsc', False) else dsg
    xmk = np.concatenate([np.ones_like(phe), xmk], 1)

    if div < N:
        # for NNT
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
            dtreg(xKT, yKT, xKE, yKE)   # Decision Tree
        ])
        # NULL model
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

    # train the network, create it if necessary
    # use xT, yT to train the network
    # use xV, yV to help decide early stop
    # use xE, yE to evaluate the generalization performance
    # append the report to 'bmk'

    kwd = dict((k, v) for k, v in kwd.items() if v is not None)
    return Bunch(kwd)
