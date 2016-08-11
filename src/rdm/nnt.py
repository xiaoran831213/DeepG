from .hlp import parms, T


class Nnt(list):
    """
    Generic layer of neural network
    """
    def __init__(self, tag=None, **kwd):
        """
        Initialize the neural network base object.
        tag: a short description of the network.
        """
        self.tag = tag
        self.__dict__.update(kwd)

    def y(self, x):
        """
        build sybolic expression of output {y} given input {x},
        which is also the defaut expression returned when the object
        is called as a function
        """
        return x

    def __call__(self, x):
        """
        makes the network a callable object.
        """
        return self.__expr__(x)

    def __expr__(self, x):
        """
        build symbolic expression of output given input. x is supposdly
        a tensor object.
        """
        return x

    def p(self):
        """
        return independent parameters - the shared tensor variables in
        output {y}'s expression.
        """
        return parms(self.y(0))

    def __repr__(self):
        return '{}{}'.format(
            "" if self.tag is None else self.tag,
            super(Nnt, self).__repr__())


def test_nnt():
    # local test
    nnt = Nnt(tag='test')
    d = T.matrix('x')
    e = nnt(d)
    p = parms(nnt)

    print(d, e, p)
