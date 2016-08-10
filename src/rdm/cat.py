import hlp
from nnt import Nnt


class Cat(Nnt):
    """
    Neural networks formed by concatinating sub-networks.
    """
    def __init__(self, nnts):
        """
        Initialize the super neural network by a list of sub networks.

        -------- parameters --------
        nnts: sub networks
        """
        super(Cat, self).__init__()

        # first dimension
        dim = [nnts[0].dim[0]]

        for p, q in zip(nnts[:-1], nnts[1:]):
            if p.dim[-1] != q.dim[0]:
                raise Exception('dimension unmatch: {} to {}'.format(p, q))
            dim.append(q.dim[0])

        # last dimension
        dim.append(nnts[-1].dim[-1])

        self.extend(nnts)
        self.dim = dim

    def y(self, x):
        """
        build sybolic expression of layer output {y} given input {x},
        which is also the defaut expression returned when the object
        is called as a function
        """
        y = x
        for net in self:
            y = net.y(y)
        return y


def test_cat():
    import numpy as np
    from os import path as pt
    from lyr import Lyr
    hlp.set_seed(120)
    x = np.load(pt.expandvars('$AZ_IMG1/lh001F1.npz'))['vtx']['tck']
    d = x.shape[1]
    x = hlp.rescale01(x)

    dim = [d/1, d/2, d/4]
    ns = [Lyr(dim=(i, j)) for i, j in zip(dim[:-1], dim[1:])]
    return x, Cat(ns)

if __name__ == '__main__':
    pass
