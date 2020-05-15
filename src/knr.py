import numpy as np
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from numpy.linalg import inv
from numpy.linalg import solve as slv
from numpy.linalg import lstsq as lsq


class Knr:

    def __init__(self, alpha=1, kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, sample_weight=None):
        """Fit Kernel Ridge regression model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data
        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values
        sample_weight : float or array-like of shape [n_samples]
            Individual weights for each sample, ignored if None is passed.
        Returns
        -------
        self : returns an instance of self.
        """
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"), multi_output=True,
                         y_numeric=True)
        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = check_array(sample_weight, ensure_2d=False)

        K = self._get_kernel(X)
        alpha = np.atleast_1d(self.alpha)

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        copy = self.kernel == "precomputed"
        # self.dual_coef_ = _solve_cholesky_kernel(K, y, alpha,
        #                                          sample_weight,
        #                                          copy)
        self.dual_coef_ = slv(K, y)
        # self.dual_coef_ = lsq(K, y)[0]
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        # test
        # coef1 = inv(K).dot(y)
        # coef2 = slv(K, y)

        self.X_fit_ = X

        return self

    def predict(self, X):
        """Predict using the kernel ridge model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.
        Returns
        -------
        C : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        check_is_fitted(self, ["X_fit_", "dual_coef_"])
        K = self._get_kernel(X, self.X_fit_)
        return np.dot(K, self.dual_coef_)
