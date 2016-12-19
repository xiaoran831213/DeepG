import theano
from time import time as tm
from hlp import S
from bas import BasicTrainer
from snp import Snap

FX = theano.config.floatX


class CmnTnr(BasicTrainer, Snap):
    """
    Class for neural network training.
    """

    def __init__(self, nnt, x=None, z=None, u=None, v=None, **kwd):
        """
        Constructor.
        : -------- parameters -------- :
        nnt: an expression builder for the neural network to be trained,
        could be a Nnt object.

        x: the inputs, with the first dimension standing for sample units.
        If unspecified, the trainer will try to evaluate the entry point and
        cache the result as source data.
        
        z: the labels, with the first dimension standing for sample units.
        if unspecified, a simi-unsupervied training is assumed as the labels
        will be identical to the inputs.

        u: the valication data inputs
        v: the validation data labels

        : -------- kwd: keywords -------- :
        -- bsz: size of a training batch
        -- lrt: basic learning rate
        -- lmb: weight decay factor, the lambda

        -- err: expression builder for the computation of training error
        between the network output {y} and the label {z}. the expression
        must evaluate to a scalar.

        -- reg: expression builder for the computation of weight panalize
        the vector of parameters {w}, the expression must evaluate to a
        scalar.

        -- mmt: momentom of the trainer

        -- vdr: validation disruption rate
        """
        # learning rate changes
        acc = kwd.get('acc', 1.04)  # acceleration
        dec = kwd.get('dec', 0.85)  # deceleration
        self.acc = S(acc, 'ACC')
        self.dec = S(dec, 'DEC')

        # supremum of gradient
        self.gsup = S(.0)

    def __onep__(self):
        """ called on new epoch. """
        # history records
        h = self.__hist__

        # update the learning rate and suprimum of gradient
        if h[-1]['terr'] < self.snap('mter', 'l'):  # accelerate
            self.snap('mter', 's')
            self.lrt.set_value(self.lrt.get_value() * self.acc.get_value())
        else:  # slow down
            self.snap('mter', 'r')
            self.lrt.set_value(self.lrt.get_value() * self.dec.get_value())

        # update minimum validation error, also save the state
        if h[-1]['verr'] < self.snap('mver', 'l') and self.u is not self.x:
            self.snap('mver', 'l')

    def __stop__(self):
        """ return true should the training be stopped. """
        h = self.__hist__
        if h[-1]['terr'] < 5e-3:
            return True
        if h[-1]['gsup'] < 5e-7:
            return True
        if h[-1]['lrt'] < 5e-10:
            return True
        return False
   
    def tune(self, nep=1, nbt=0, rec=0, prt=0):
        """ tune the parameters by running the trainer {nep} epoch.
        nep: number of epoches to go through
        nbt: number of extra batches to go through

        rec: frequency of recording. 0 means record after each epoch,
        which is the default, otherwise, record for each batch, which
        can be time consuming.

        prt: frequency of printing.
        """
        bt = self.bt
        b0 = bt.eval().item()  # starting batch
        ei, bi = 0, 0  # counted epochs and batches

        nep = nep + nbt // self.nbat()
        nbt = nbt % self.nbat()

        while (ei < nep or bi < nbt) and not self.__stop__():
            # send one batch for training
            t0 = tm()
            self.step()
            self.__time__ += tm() - t0

            # update history
            bi = bi + 1  # batch count increase by 1
            if rec > 0:  # record each batch
                self.__hist__.append(self.__rpt__())
            if prt > 0:  # print each batch
                self.cout()

            # see the next epoch relative to batch b0?
            if bt.get_value().item() == b0:
                ei = ei + 1  # epoch count increase by 1
                bi = 0  # reset batch count

            # after an epoch
            if bt.get_value().item() == 0:
                # record history
                if rec == 0:  # record at new epoch
                    self.__hist__.append(self.__rpt__())
                # print
                if prt == 0:  # print
                    self.cout()

                # adjustment at the end of each epoch
                if self.__onep__:
                    self.__onep__()
