import numpy as np


def roc_one(x, z, plot=0):
    n_p = (x > 0).sum()         # number of positives
    n_n = x.size - n_p          # number of negatives
    t_p = np.ndarray((100,))    # true positives (se)
    f_p = np.ndarray((100,))    # false positives (1-sp)

    for i, t in enumerate(np.linspace(1, 0, 100)):
        t_p[i] = ((z > t) & (x > 0)).sum() / n_p
        f_p[i] = ((z > t) & (x < 1)).sum() / n_n

    if (n_p == 0):
        t_p[:] = 0
    if (n_n == 0):
        f_p[:] = 1
    
    # get AUC
    auc = t_p.mean()

    # plot?
    if plot:
        import matplotlib.pyplot as pl
        pl.plot(f_p, t_p)
        pl.show()
    return {'roc': (f_p, t_p), 'auc': auc}


def roc_all(x, z, plot=0):
    n = x.shape[0]
    rcs = np.ndarray((n, 2, 100), dtype='f4')
    acs = np.ndarray(n, dtype='f4')

    for i in range(n):
        r = roc_one(x[i, ], z[i, ])
        rcs[i, 0, :] = r['roc'][0]
        rcs[i, 1, :] = r['roc'][1]
        acs[i] = r['auc']

    return {'roc': rcs, 'auc': acs}

    
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


def hist(x, title='x_histogram'):
    import matplotlib.pyplot as pl
    pl.hist(x)
    pl.title(title)
    pl.xlabel("x")
    pl.ylabel("f")
    pl.show()
