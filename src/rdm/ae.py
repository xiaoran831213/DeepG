import numpy as np

import hlp
from cat import Cat
from lyr import Lyr


class AE(Cat):
    """
    Basic auto-encoder
    """

    def __init__(self,
                 dim,
                 ec_w=None,
                 ec_b=None,
                 ec_s=None,
                 dc_w=None,
                 dc_b=None,
                 dc_s=None):
        """
        Initialize the denosing auto encoder by specifying the the dimension
        of the input  and output.
        The constructor also receives symbolic variables for the weights and
        bias.

        -------- parameters --------
        dim: a 2-tuple of input/output dimensions
        dim[0] is the dimension of the input (visible units)
        dim[1] is the dimension of the code (hidden units)

        ec_w: (optional) encoder weight of dimension (d_1, d_2),
        by default it is randomly initialized.

        ec_b: (optional) encoder bias of dimension d_2,
        by default the initial value is 0.

        ec_s: (optional) encoder nonlinearity. The default is Sigmoid.
        Specify 1 to disable nonlinearity

        dc_w: (optional) decoder weight of dimension (d_2, d_1), by default
        it is constrained to be the transpose of encoding weight

        dc_b: (optional) decoder bias of dimension d_1, by default the initial
        value is 0.
        dc_s: (optional) decoder nonlinearity, by default be the same with the
        encoder.
        """
        """ the encoder view of the autoencoder """
        ec = Lyr(dim, ec_w, ec_b, ec_s)
        """ the decoder view of the autoencoder """
        # dimension of the decoder
        dim = [dim[1], dim[0]]

        # if the decoding weight ec_w is None, it is contrainted to transposed
        # encoding wight ec_w'
        dc_w = ec.w.T if dc_w is None else dc_w

        # if the decoding nonlinear is None, the same nonlinear of encoding is
        # used
        dc_s = ec.s if dc_s is None else dc_s
        dc = Lyr(dim, dc_w, dc_b, dc_s)

        # the default view is a concatinated network of dimension d0, d1, d0
        super(AE, self).__init__([ec, dc])
        self.ec = ec
        self.dc = dc


def test_ae1():
    import os.path as pt

    hlp.set_seed(120)

    x = np.load(pt.expandvars('$AZ_SP1/lh001F1.npz'))['vtx']['tck']
    d = x.shape[1]
    x = hlp.rescale01(x)

    dim = [d / 1, d / 2]
    m = AE(dim=dim)

    return x, m


if __name__ == '__main__':
    pass
