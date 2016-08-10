import numpy as np
from . import hlp
from .hlp import S
from .hlp import T
from .nnt import Nnt


class Pcp(Nnt):
    """
    A Perceptron, which is the full linear recombination of the input
    elements and a bias(or intercept), followed by an per-element non-
    linear transformation(usually sigmoid).
    """
    def __init__(self, dim, w=None, b=None, s=None, tag=None, **kwd):
        """
        Initialize the perceptron by specifying the the dimension of input
        and output.
        The constructor also receives symbolic variables for the input,
        weights and bias. Such a symbolic variables are useful when, for
        example, the input is the result of some computations, or when
        the weights are shared between the layers

        -------- parameters --------
        dim: a 2-tuple of input/output dimensions

        w: (optional) weight of dimension (d_1, d_2), which is randomly
        filled by default.
        d_1 specify the input dimension
        d_2 specify the output dimension

        b: (optional) bias of dimension d_2, it is zero filled by default

        s: (optional) nonlinear tranformation of the weighted sum.
        By default the sigmoid function is used.
        To suppress nonlinearity, specify 1 instead.
        """
        super(Pcp, self).__init__(tag, **kwd)

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
                dtype=hlp.FX())
            w = S(w, 'w')

        if b is None:
            b = np.zeros(dim[1], dtype=hlp.FX())
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
        """ build symbolic expression of {y} given {x}.
        For a perceptron, it is the full linear recombination of the
        input elements and an bias(or intercept), followed by an
        element-wise non-linear transformation(usually sigmoid)
        """
        affin = T.dot(x, self.w) + self.b
        if self.s is 1:
            return affin
        else:
            return self.s(affin)


def test_lyr():
    from os import path as pt
    hlp.set_seed(120)
    x = np.load(pt.expandvars('$AZ_SP1/lh001F1.npz'))['vtx']['tck']
    d = (x.shape[1], x.shape[1] / 2)
    x = hlp.rescale01(x)

    nt = Pcp(dim=d)
    return x, nt


if __name__ == '__main__':
    pass
