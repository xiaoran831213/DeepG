import theano
from hlp import S
from bas import Base
from snp import Snap

FX = theano.config.floatX


class Bold(Snap, Base):
    """
    Bold Driver implementation for nerual networks.
    """
    def __init__(self, *arg, **kwd):
        """
        acc: acceleration when the trainer see an reduction in error.
        dec: deceleration when the trainer see an increase in error.

        """
        # learning rate changes
        self.acc = S(kwd.get('acc', 1.04), 'ACC')  # acceleration
        self.dec = S(kwd.get('dec', 0.85), 'DEC')  # deceleration

        # initialize super class.
        super(Bold, self).__init__(*arg, **kwd)
        self.snap('mter', 's')  # minimum training error
        self.snap('mver', 's')  # minimum validation error

    def __onep__(self):
        """ called on new epoch. """
        
        # history records
        r = self.__hist__[-1]

        # update the learning rate and suprimum of gradient
        if r['terr'] < self.snap('mter', 'l')['terr']:  # accelerate
            self.lrt.set_value(self.lrt.get_value() * self.acc.get_value())
        else:                   # slow down, and restore saved state
            self.snap('mter', 'r')
            self.lrt.set_value(self.lrt.get_value() * self.dec.get_value())

        self.snap('mter', 's')

        # update minimum validation error, also save the state
        if r['verr'] < self.snap('mver', 'l')['verr'] and self.u is not self.x:
            self.snap('mver', 's')

        # super class call
        super(Bold, self).__onep__()
