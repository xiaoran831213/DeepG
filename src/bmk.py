import numpy as np
from itertools import product as cpd, chain as cat
import pandas as pd


# ---------- benchmark references ---------- #
rec = np.dtype([
    ('mtd', 'U32'), ('par', 'U32'),
    ('err', '<f4'), ('cor', '<f4'), ('auc', '<f4')])


def nul(xT, yT, xV=None, yV=None, fam='gau'):
    """ null predictor. """
    print('benchmark:', 'NUL')
    if xV is None or yV is None:
        xV, yV = xT, yT

    # testing
    ret = []
    yH = np.repeat(yV.mean(), yV.size)

    if fam == 'bin':
        err = np.nanmean(-yV * np.log(yH) - (1 - yV) * np.log(1 - yH))
    else:
        err = np.mean((yH - yV) ** 2)
    ret.append(dict(mtd='nul', par='-', key='ERR', val=err))

    ret = pd.DataFrame(ret)
    return ret


def knreg(xT, yT, xV=None, yV=None):
    """ Kernel Regression. """
    from sklearn.kernel_ridge import KernelRidge as KR
    # from knr import Knr as KR
    print('benchmark:', 'KNR')
    if xV is None or yV is None:
        xV, yV = xT, yT

    # fitting
    # alpha, kernel, and degree
    a = [.0]  # , .1, .2, .5, 1., 2., 5.]
    k = ['linear', 'rbf', 'laplacian', 'sigmoid', 'cosine']
    cfg = cat(cpd(a, k, [None]), cpd(a, ['poly'], [1, 2, 3]))
    ret = list()
    for a, k, d in cfg:
        print(a, k, d)
        clf = KR(alpha=a, kernel=k, degree=d)
        clf.fit(xT, yT)
        # testing
        yH = clf.predict(xV)
        err = np.mean((yH - yV) ** 2)
        cor = np.corrcoef(np.ravel(yV), np.ravel(yH))[0, 1]

        # report
        par = k
        if a > 0.0:
            par = 'a={.1f}, {}'.format(a, par)
        if d is not None:
            par = '{}({:d})'.format(par, d)
        ret.append(dict(mtd='knr', par=par, key='ERR', val=err))
        ret.append(dict(mtd='knr', par=par, key='COR', val=cor))

    ret = pd.DataFrame(ret)
    return ret


def svreg(xT, yT, xV=None, yV=None):
    """ Support Vector Regression. """
    from sklearn import svm
    print('benchmark:', 'SVR')
    if xV is None or yV is None:
        xV, yV = xT, yT

    # fitting
    k = ['linear', 'rbf', 'sigmoid']
    cfg = cat(cpd(k, [1]), cpd(['poly'], [1, 2, 3]))
    ret = list()
    for k, d in cfg:
        clf = svm.SVR(kernel=k, degree=d, C=1e12)
        clf.fit(xT, np.ravel(yT))

        # testing
        yH = clf.predict(xV)
        err = np.mean((yH - yV) ** 2)
        cor = np.corrcoef(np.ravel(yV), np.ravel(yH))[0, 1]
        par = k
        if k == 'poly':
            par = '{}({:d})'.format(par, d)
        ret.append(dict(mtd='svr', par=par, key='ERR', val=err))
        ret.append(dict(mtd='svr', par=par, key='COR', val=cor))

    ret = pd.DataFrame(ret)
    return ret


def dtreg(xT, yT, xV=None, yV=None):
    """ Decision Tree Regression. """
    from sklearn.tree import DecisionTreeRegressor as DTR

    print('benchmark:', 'DTR')
    dpt = [1, 2, 3, 4, 5, 10]
    ret = list()
    for par in dpt:
        # fitting
        dtr = DTR(splitter='best', max_depth=par)
        dtr.fit(xT, np.ravel(yT))

        # testing
        yH = dtr.predict(xV)
        err = np.mean((yH - yV) ** 2)
        cor = np.corrcoef(np.ravel(yV), np.ravel(yH))[0, 1]

        # report
        ret.append(dict(mtd='dtr', par=par, key='ERR', val=err))
        ret.append(dict(mtd='dtr', par=par, key='COR', val=cor))

    ret = pd.DataFrame(ret)
    return ret


def svclf(xT, yT, xV=None, yV=None):
    """ Support Vector Classifier (binary). """
    from sklearn import svm
    from sklearn.metrics import roc_auc_score as AUC

    print('benchmark:', 'SVM')
    kms = ['linear', 'poly', 'rbf', 'sigmoid']
    ret = list()
    yT, yV = np.ravel(yT), np.ravel(yV)
    for par in kms:
        # fitting
        clf = svm.SVC(kernel=par, probability=True)
        clf.fit(xT, yT)

        # testing
        yH = clf.predict_proba(xV)[:, 1]
        err = np.nanmean(-yV * np.log(yH) - (1 - yV) * np.log(1 - yH))
        cor = np.corrcoef(np.ravel(yV), np.ravel(yH))[0, 1]
        auc = AUC(np.ravel(yV), np.ravel(yH))
        ret.append(('svm', par, err, cor, auc))
    ret = np.array(ret, rec)
    return ret


def dtclf(xT, yT, xV=None, yV=None):
    """ decision tree classifier (binary). """
    from sklearn import tree
    from sklearn.metrics import roc_auc_score as AUC

    print('benchmark:', 'DTC')
    dpt = [2, 3, 5, 10, None]
    ret = list()
    yT, yV = np.ravel(yT), np.ravel(yV)
    for par in dpt:
        # fitting
        clf = tree.DecisionTreeClassifier(max_depth=par)
        clf.fit(xT, yT)

        # testing
        yH = np.clip(clf.predict_proba(xV)[:, 1], 1e-10, 1-1e-10)
        err = np.nanmean(-yV * np.log(yH) - (1 - yV) * np.log(1 - yH))
        cor = np.corrcoef(yV, yH)[0, 1]
        auc = AUC(yV, yH)
        ret.append(('dtc', par, err, cor, auc))

    ret = np.array(ret, rec)
    return ret
