import numpy as np
from numpy.random import normal, choice
import itertools as it


def sim(gmx, **kwd):
    """ simulate phenotype frsom a segment of genome.
    ** frq: frequency of functional variable

    ** rsq: r-square for gaussian family, the deterministic coeficient.
    for non-gaussian family, this serves as a noise controller.

    ** fam: distribution family of simulated responses
    -- gau: gaussian
    -- sin: sin
    -- bin: binomial

    ** mdl: the underlying linear model, and base expansions
    1) variable codings:
    --   a: additve
    --   g: mix of additve, dominence, and recessive
    --   p: the parity base, controlled by {fpr}
    2) base expansions:
    -- g:g: cross product terms
    -- g^2: squared ter
    -- g*g: all terms up to the 2nd order (g + g:g + g^2).

    ** fpr: proportion of parity terms with in functional variants.
    it is 1 by defaut.
    """
    # handle genomic matrix
    gmx = gmx.sum(1)            # get dosage values
    N, P = gmx.shape[0], gmx.shape[1]

    # addtive, dominent, and ressesive marks
    mdl = kwd.get('mdl', 'a')
    if 'g' in mdl:
        print('SIM: generate dominent and ressesive bases.')
        m = np.random.choice(3, P)
        gmx[:, m == 1] = (gmx[:, m == 1] > 0).astype('f') * 2  # dom
        gmx[:, m == 2] = (gmx[:, m == 2] > 1).astype('f') * 2  # res

    # remove duplicate columns
    gmx = np.unique(gmx, axis=1)
    if gmx.shape[1] < P:
        print('SIM: drop', P - gmx.shape[1], 'duplicate bases.')
        P = gmx.shape[1]

    # remove non-informative bases
    gmx = gmx[:, gmx.std(0) > 0]
    if gmx.shape[1] < P:
        print('SIM: drop', P - gmx.shape[1], 'null-info bases.')
        P = gmx.shape[1]

    # pick functional variants
    P = int(gmx.shape[1] * kwd.get('frq', 0.25))
    gmx = gmx[:, np.random.permutation(range(gmx.shape[1]))[:P]]

    # base expansion
    if ':' in mdl or '*' in mdl:  # crossover
        cmb = it.combinations(range(P), 2)
        gmx = np.array([gmx[:, i] * gmx[:, j] for i, j in cmb]).T
    if '^' in mdl or '*' in mdl:  # full 2nd order
        cmb = it.combinations_with_replacement(range(P), 2)
        emx = np.array([gmx[:, i] * gmx[:, j] for i, j in cmb]).T
        gmx = np.hstack([gmx, emx])
    if '^' in mdl:              # squared
        gmx = gmx ** 2
    if 'p' in mdl:              # parity
        fpr = kwd.get('fpr', 1)
        if fpr is not None:
            fpr = int(P * fpr) if fpr < 1 else int(fpr)
            idx = gmx[:, choice(P, fpr)].sum(1).astype('<i4') % 2
            gmx = np.array([1, -1])[idx, np.newaxis]
    P = gmx.shape[1]

    # remove uninformative bases
    gmx = gmx[:, gmx.std(0) > 0]
    if gmx.shape[1] < P:
        print('SIM: drop', P - gmx.shape[1], 'null-info base expansion.')
        P = gmx.shape[1]

    # remove duplicated bases
    gmx = np.unique(gmx, axis=1)
    if gmx.shape[1] < P:
        print('SIM: drop', P - gmx.shape[1], 'duplicate base expansion.')
        P = gmx.shape[1]

    # weights drawn from Gaussian
    w = normal(size=[P, 1])

    # simulate signal
    xw = np.dot(gmx, w)         # linear signal

    # mix with noise
    rsq = kwd.get('rsq', 0.30)
    if rsq > 0.0:
        rs = np.sqrt((1 - rsq) / rsq) * np.std(xw)
        eta = xw + np.random.normal(0.0, rs, [N, 1])  # signal + noise
    else:
        print('null effect simulated.')
        eta = np.random.normal(0.0, np.std(xw), [N, 1])

    # put through link function
    fam = kwd.get('fam', 'gau')
    if fam == 'sin':
        phe = np.sin(eta)
    elif fam == 'bin':
        mu = rsq/(1 + np.exp(-xw)) + (1 - rsq) * 0.5
        phe = np.random.binomial(1, mu).reshape(N, 1)
    else:
        phe = eta
    phe = phe.astype('f')

    return phe
