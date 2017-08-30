import numpy as np


# ---------- benchmark references ---------- #
rec = np.dtype([
    ('mtd', 'U32'), ('par', 'U32'),
    ('err', '<f4'), ('cor', '<f4'), ('auc', '<f4')])


def knreg(xT, yT, xV=None, yV=None):
    """ Kernel Regression. """
    from sklearn.kernel_ridge import KernelRidge as KR
    print('benchmark:', 'DTR')
    if xV is None or yV is None:
        xV, yV = xT, yT

    # fitting
    kms = ['linear', 'poly', 'rbf', 'laplacian', 'sigmoid', 'cosine']
    ret = list()
    for par in kms:
        clf = KR(alpha=1.0, kernel=par)
        clf.fit(xT, yT)

        # testing
        yH = clf.predict(xV)
        err = np.mean((yH - yV) ** 2)
        cor = np.corrcoef(np.ravel(yV), np.ravel(yH))[0, 1]
        ret.append(('knr', par, err, cor, np.nan))
    ret = np.array(ret, rec)
    return ret


def svreg(xT, yT, xV=None, yV=None):
    """ Support Vector Regression. """
    from sklearn import svm
    print('benchmark:', 'SVR')
    if xV is None or yV is None:
        xV, yV = xT, yT

    # fitting
    kms = ['linear', 'poly', 'rbf', 'sigmoid']
    ret = list()
    for par in kms:
        clf = svm.SVR(kernel=par)
        clf.fit(xT, np.ravel(yT))

        # testing
        yH = clf.predict(xV)
        err = np.mean((yH - yV) ** 2)
        cor = np.corrcoef(np.ravel(yV), np.ravel(yH))[0, 1]
        ret.append(('svr', par, err, cor, np.nan))
    ret = np.array(ret, rec)
    return ret


def dtreg(xT, yT, xV=None, yV=None):
    """ Decision Tree Regression. """
    from sklearn.tree import DecisionTreeRegressor as DTR

    print('benchmark:', 'DTR')
    dpt = [2, 3, 5, 10, 20, 30, 50]
    ret = list()
    for par in dpt:
        # fitting
        dtr = DTR(max_depth=par)
        dtr.fit(xT, np.ravel(yT))

        # testing
        yH = dtr.predict(xV)
        err = np.mean((yH - yV) ** 2)
        cor = np.corrcoef(np.ravel(yV), np.ravel(yH))[0, 1]
        ret.append(('dtr', par, err, cor, np.nan))
    ret = np.array(ret, rec)
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
    dpt = [2, 3, 5, 10, 20, 30, 50]
    ret = list()
    yT, yV = np.ravel(yT), np.ravel(yV)
    for par in dpt:
        # fitting
        clf = tree.DecisionTreeClassifier(max_depth=par)
        clf.fit(xT, yT)

        # testing
        yH = clf.predict_proba(xV)[:, 1]
        err = np.nanmean(-yV * np.log(yH) - (1 - yV) * np.log(1 - yH))
        cor = np.corrcoef(yV, yH)[0, 1]
        auc = AUC(yV, yH)
        ret.append(('dtc', par, err, cor, auc))

    ret = np.array(ret, rec)
    return ret
