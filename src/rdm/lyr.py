import numpy as np
import theano
import hlp
from hlp import S
from hlp import T
import pdb
import nnt
from nnt import Nnt

class Lyr(Nnt):
    """
    Generic layer of neural network
    """
    def __init__(self, dim, w = None, b = None, s = None):
        """
        Initialize the neural network layer class by specifying the the dimension of the
        input, and the dimension of the output.
        The constructor also receives symbolic variables for the input, weights and bias.
        Such a symbolic variables are useful when, for example the input is the result of
        some computations, or when weights are shared between the layers
        
        -------- parameters --------
        dim: a 2-tuple of input/output dimensions

        w: (optional) weight of dimension (d_1, d_2), which is randomly filled by default.
        d_1 specify the input dimension
        d_2 specify the output dimension

        b: (optional) bias of dimension d_2, it is zero filled by default

        s: (optional) nonlinear tranformation of the weighted sum.
        By default the sigmoid function is used.
        To suppress nonlinearity, specify 1 instead.
        """
        super(Lyr, self).__init__()

        ## I/O dimensions
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
                dtype = hlp.FX())
            w = S(w, 'w')

        if b is None:
            b = np.zeros(dim[1], dtype = hlp.FX())
            b = S(b, 'b')

        self.w = w
        self.b = b

        if s is None:
            s = T.nnet.sigmoid
        self.s = s

    ## a Lyr cab be represented by the nonlinear funciton and I/O dimensions
    def __repr__(self):
        return '{}({}-{})'.format(
            str(self.s)[0].upper(), self.dim[0], self.dim[1])

    def y(self, x):
        """
        build sybolic expression of layer output {y} given input {x},
        which is also the defaut expression returned when the Lyr object
        is being called as a function
        """
        affin = T.dot(x, self.w) + self.b
        if self.s is 1:
            return  affin
        else:
            return  self.s(affin)

def test_lyr():
    from os import path as pt
    hlp.set_seed(120)
    x = np.load(pt.expandvars('$AZ_SP1/lh001F1.npz'))['vtx']['tck']
    d = (x.shape[1], x.shape[1]/2)
    x = hlp.rescale01(x)
    
    ## add project root to python path
    if not os.environ['AZ_PRJ'] in sys.path:
        sys.path.insert(0, os.environ['AZ_PRJ'])
    import src.hlp as rut_hlp
    
    nt = Lyr(dim=d)
    return x, nt

if __name__ == '__main__':
    pass
