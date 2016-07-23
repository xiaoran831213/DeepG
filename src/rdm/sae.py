import numpy as np
import hlp
import cat
from cat import Cat
from ae import AE
import pdb

class SAE(Cat):
    """
    Stacked Auto Encoder
    """
    def __init__(self, AEs):
        """
        Initialize the stacked auto encoder by a list of code dimensions.
        The weight and bias terms in the AEs are initialized by default rule.
        -------- parameters --------
        AEs: a list of autoencoders
        """
        
        """ the default view of the stacked autoencoders"""
        sa = AEs
        
        """ the encoder view of the stacked autoencoders """
        ec = cat.Cat([a.ec for a in sa])

        """ the decoder view of the stacked autoencoders """
        dc = cat.Cat([a.dc for a in reversed(sa)])

        self.sa = sa            # default view
        self.ec = ec            # encoder view
        self.dc = dc            # decoder view

        nts = []
        nts.extend(ec)
        nts.extend(dc)
        super(SAE, self).__init__(nts)
        
    @staticmethod
    def from_dim(dim):
        """ create SAE by specifying encoding dimensions
        dim: a list of encoding dimensions
        """
        AEs = [AE(d) for d in zip(dim[:-1], dim[1:])]
        return SAE(AEs)

    def sub(self, depth, start = None, copy = False):
        """ get sub stack from of lower encoding depth
        -------- parameters --------
        depth: depth of the sub-stack, should be less then the full
        encoder
        
        start: starting level of sub-stack extraction. by default
        always extract from the lowest level.

        copy: parameters in the sub stack is deeply copied from the
        full stack
        """
        ret = SAE(self.sa[start:depth])
        if copy:
            import copy
            ret = copy.deepcopy(ret)
        return ret
        
def test_sa1():
    import os.path as pt
    hlp.set_seed(120)

    x = np.load(pt.expandvars('$AZ_SP1/lh001F1.npz'))['vtx']['tck']
    d = x.shape[1]
    x = hlp.rescale01(x)

    dim = [d/1, d/2, d/4, d/8, d/16]
    m = SAE.from_dim(dim)
    return x, m

if __name__ == '__main__':
    pass
