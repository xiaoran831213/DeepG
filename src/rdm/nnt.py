from .hlp import parms
from .hlp import is_tnsr


class Nnt(list):
    """
    Generic layer of neural network
    """
    def __init__(self, tag=None):
        """
        Initialize the neural network base object.
        tag: a short description of the network.
        """
        self.tag = tag

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
        if is_tnsr(x):
            return self.__expr__(x)
        else:
            return lambda u: self.__expr__(x(u))

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
