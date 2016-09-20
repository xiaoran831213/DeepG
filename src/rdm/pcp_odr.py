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
    def __init__(self, dim, lvl, w=None, b=None, s=None, tag=None, **kwd):
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

        w: (optional) weight matrix (p, q), randomly filled by default.
        b: (optional) bias matrix (p, k), zero filled by default.

        s: (optional) nonlinear tranformation of the weighted sum.
        By default the sigmoid function is used.
        To suppress nonlinearity, specify 1 instead.
        """
        super(PcpOdr, self).__init__(tag, **kwd)

        # I/O dimensions
        self.dim = dim

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
            b = np.zeros((lvl-1, 1, dim[1]), dtype='float32')
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

        1) let x.dim = (n, p), w.dim = (p, q), and b.dim = (k, 1, q)
        let a = x.dot(self.w) + self.b, and a.dim = (k, n, q)
        
        n: number of observations
        p: dimension of input per observation
        q: dimension of output per observation
        k: number of ordinal levels

        logit(p(y<=j)) = b[j] + sum(x[i,] * w[,i]),
        0 <= i < n, 0 <= j < k-1
        --> p(y <=   j) = sigmoid{b[j] + sum(x[i,] * w[,i])}
        --> and p(y <= k-1) = 1
        """
        c = self.s(x.dot(self.w) + self.b)
        return self.__prob__(c)

    def __prob__(self, c):
        """ build symbolic expression for the k probabilities of each
        level, given the k-1 cumulative probability.
        """
        return p
        

if __name__ == '__main__':
    pass
