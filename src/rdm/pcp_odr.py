import numpy as np
from . import hlp
from .hlp import S
from .hlp import T
from .nnt import Nnt


class PcpOdr(Nnt):
    """
    A Perceptron, which is the full linear recombination of the input
    elements and a bias(or intercept), followed by an per-element non-
    linear transformation(usually sigmoid).
    The ordinal percepptron takes input of discrate level (e.g. poor,
    fair, normal, fine, exceptional).
    """

    def __init__(self, dim, lvl, mod=None, w=None, b=None, s=None, **kwd):
        """
        Initialize the perceptron by specifying the the dimension of input
        and output.
        The constructor also receives symbolic variables for the input,
        weights and bias. Such a symbolic variables are useful when, for
        example, the input is the result of some computations, or when
        the weights are shared between the layers

        -------- parameters --------
        dim: the dimension of input (p) and output (q) per sample.
        dim = (p, q)

        lvl: the number of levels of input
        mod: mode of transformation:
        0 --> from observed levels to hidden units.
        1 --> from hidden units to predicted probabilities of levels.

        w: (optional) weight matrix (p, q), randomly filled by default.
        b: (optional) bias matrix (p, k), zero filled by default.

        s: (optional) nonlinear tranformation of the weighted sum.
        By default the sigmoid function is used.
        To suppress nonlinearity, specify 1 instead.
        """
        super(PcpOdr, self).__init__(**kwd)

        # I/O dimensions
        self.dim = dim

        # output levels
        self.lvl = lvl

        # transformation mode
        self.mod = mod

        # transformation direction
        self.dir = dir

        # note : W' was written as `W_prime` and b' as `b_prime`
        """
        # W is initialized with `initial_W` which is uniformely sampled
        # from -4*sqrt(6./(n_vis+n_hid)) and
        # 4 * sqrt(6./(n_hid+n_vis))the output of uniform if
        # converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        """
        if w is None:
            w = np.asarray(
                hlp.rs_np.uniform(
                    low=-4 * np.sqrt(6. / (dim[0] + dim[1])),
                    high=4 * np.sqrt(6. / (dim[0] + dim[1])),
                    size=dim),
                dtype='float32')
            w = S(w, 'w')

        if b is None:
            if mod is 1:
                b = np.zeros(dim[-1], dtype='float32')
            elif mod is 2:
                b = np.zeros((lvl - 1, 1, dim[-1]), dtype='float32')
            else:
                b = np.linspace(0, 1, lvl, dtype='float32')[0:lvl - 1]
                b = b.repeat(dim[-1]).reshape(lvl - 1, 1, dim[-1])

        b = S(b, 'b')

        self.w = w
        self.b = b

        if s is None:
            s = T.nnet.sigmoid
        self.s = s

    # a Pcp cab be represented by the nonlinear funciton and I/O dimensions
    def __repr__(self):
        return '{}({}X{})'.format(
            str(self.s)[0].upper(), self.dim[0], self.dim[1])

    def __expr__(self, x):
        """ build symbolic expression of k probabilities p[0], p[1], ...,
        p[k-1]} for the output levels, given p dimensional input {x} and
        q dimensional output, which is done for all {n} observations.

        1) x.dim = (n, p), w.dim = (p, q), and b.dim = (k, 1, q)
        n: number of observations
        p: dimension of input
        q: dimension of output
        r: number of levels of the output
        s: a link function to transform a real number into a propability.

        p(y[i,j] <= l) = s{b[l, *, j] + sum(x[i,k] * w[k,j], k = 0, .. p-1)}
        p(y <= k) = 1,
        where i = 0 .. n indexes observations, j = 0 .. p-1 indexes input
        features x[j], k = 0 .. q indexes output feature y[k], and l = 0 ...
        r-1 indexes the output levels.
        """
        if self.mod is 1:       # treat input additively
            v = np.linspace(0, 1, self.lvl, dtype='f4').reshape(self.lvl, 1, 1)
            _ = (v * x)
            return self.s(T.dot(_.sum(0), self.w) + self.b)
        elif self.mod is 2:     # treat input as indicators
            v = np.linspace(0, 1, self.lvl, dtype='f4').reshape(self.lvl, 1, 1)
            _ = (v * x)
            b = T.concatenate([self.b, T.zeros_like(self.b[0:1])])
            return self.s(T.sum(T.dot(_, self.w) + b, 0))
        else:                   # treat output
            _ = self.s(T.dot(x, self.w) + self.b)
            return T.concatenate([_[0:1], _[1:] - _[0:-1], 1 - _[-1:]])

    def CE(y, z):
        """ build symbolic expression of multinomial cross entrophy given the
        expected outcome {z}, and the predicted probabilities {y}.
        In reality, the mean negative log likelihood across observation is
        to be returned.
        y.dim = (..., k, n, q) = z.dim
        """
        return -(z * T.log(y)).sum((0, -1)).mean()

    def C1(y, z, lvl):
        """ build symbolic expression of binary cross entrophy. """
        v = np.linspace(0, 1, lvl).reshape(lvl, 1, 1)
        y = (y * v).sum(0)
        z = (z * v).sum(0)
        return -(z * T.log(y) + (1 - z) * T.log(1 - y)).sum(-1).mean()


if __name__ == '__main__':
    pass


def test_pdr():
    """ """
    x = np.random.uniform(0, 1, 1750 * 256).reshape(1750, 256)
    d = x.shape
    w = np.random.uniform(
        low=-4 * np.sqrt(6. / (d[0] + d[1])),
        high=4 * np.sqrt(6. / (d[0] + d[1])),
        size=(256, 512))
    b = np.repeat([-1, 1], 512).reshape(2, 1, 512)
    return x, w, b
