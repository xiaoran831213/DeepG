import hlp
import pdb

class Nnt(list):
    """
    Generic layer of neural network
    """
    def __init__(self):
        """
        Initialize the neural network base object.
        """
        self.tag = None

    def y(self, x):
        """
        build sybolic expression of output {y} given input {x},
        which is also the defaut expression returned when the object
        is called as a function
        """
        return  x

    def __call__(self, x):
        """
        build symbolic expression of output given input. This makes the
        object callable.
        """
        return self.y(x)
        
    def p(self):
        """
        return independent parameters - the shared tensor variables in
        output {y}'s expression.
        """
        return hlp.parms(self.y(0))

    def __repr__(self):
         return '{}{}'.format(
             "" if self.tag is None else self.tag,
             super(Nnt, self).__repr__())
