import numpy as np
import theano
from theano import tensor as T
from theano import function as F
import sys
try:
    sys.path.extend(['..'] if '..' not in sys.path else [])
    from hlp import S, C, parms, paint
except ValueError as e:
    from hlp import S, C, parms, paint

from time import time as tm
import sys
from matplotlib import pyplot as plt
from pdb import set_trace
FX = theano.config.floatX


def __err_CE__(y, z):
    """ symbolic expression of cross entrophy
    y: produced output
    z: expected output

    The first dimension denote unit of sampling
    """
    return -(z * T.log(y) + (1 - z) * T.log(1 - y))


def __err_L2__(y, z):
    """ symbolic expression of __err_L2__ norm
    y: produced output
    z: expected output

    The first dimension denote unit of sampling
    """
    return T.sqrt((y - z)**2)


def __err_L1__(y, z):
    """ symbolic expression of __err_L1__ norm
    y: produced output
    z: expected output

    The first dimension of y, z denote the batch size.
    """
    return abs(y - z)

__errs__ = {
    None: __err_CE__,
    'CE': __err_CE__,
    'L1': __err_L1__,
    'L2': __err_L2__}


def __reg_L0__(x, thd=1e-06):
    """ build expression of L0 norm given a vector. """
    return T.sum((T.abs_(x) > thd), dtype=FX)


def __reg_L1__(x):
    """ build expression of __err_L1__ norm given a vector. """
    return T.sum(T.abs_(x))


def __reg_L2__(x):
    """ build expression of __err_L2__ norm given a vector. """
    return T.sqrt(T.sum(x**2))

__regs__ = {
    None: __reg_L1__,
    'L0': __reg_L0__,
    'L1': __reg_L1__,
    'L2': __reg_L2__}


class Trainer(object):
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
        # numpy random number generator
        seed = kwd.pop('seed', None)
        nrng = kwd.pop('nrng', np.random.RandomState(seed))

        from theano.tensor.shared_randomstreams import RandomStreams
        trng = kwd.pop('trng', RandomStreams(nrng.randint(0x7FFFFFFF)))

        # private members
        self.__seed__ = seed
        self.__nrng__ = nrng
        self.__trng__ = trng
        
        # expression of error and regulator terms
        err = __errs__[kwd.get('err')]
        reg = __regs__[kwd.get('reg')]

        # the validation disruption
        self.vdr = S(kwd.get('vdr'), 'VDR')

        # current epoch index, use int64
        self.ep = S(0, 'EP')

        # training batch ppsize, use int64
        bsz = kwd.get('bsz', 20)
        self.bsz = S(bsz, 'BSZ')

        # current batch index, use int64
        self.bt = S(0, 'BT')

        # momentumn, make sure momentum is a sane value
        mmt = kwd.get('mmt', .0)
        self.mmt = S(mmt, 'MMT')

        # learning rate
        lrt = kwd.get('lrt', .01)
        lrt_inc = kwd.get('inc', 1.04)
        lrt_dec = kwd.get('dec', 0.5)
        self.lrt = S(lrt, 'LRT')
        self.lrt_inc = S(lrt_inc, 'LRT_INC')
        self.lrt_dec = S(lrt_dec, 'LRT_DEC')

        # the raio of weight decay, lambda
        lmd = kwd.get('lmd', .0)
        self.lmd = S(lmd, 'LMD')

        # the neural network
        self.nnt = nnt
        self.dim = (nnt.dim[0], nnt.dim[-1])

        # superium of gradient
        self.gsup = S(.0)

        # inputs and labels, for modeling and validation
        x = S(np.zeros((bsz * 2, self.dim[0]), 'f') if x is None else x)
        z = x if z is None else S(z)
        u = x if u is None else S(u)
        v = u if v is None else S(v)
        self.x, self.z, self.u, self.v = x, z, u, v

        # -------- construct trainer function -------- *
        # 1) symbolic expressions
        x = T.tensor(name='x', dtype=x.dtype, broadcastable=x.broadcastable)
        z = T.tensor(name='z', dtype=z.dtype, broadcastable=z.broadcastable)
        y = nnt(x)  # the symbolic batch output

        # list of independant symbolic parameters to be tuned
        pars = parms(y)  # parameters

        # list of  symbolic weights to apply decay
        wgts = [p for p in pars if p.name == 'w']         # weights
        vwgt = T.concatenate([w.flatten() for w in wgts])  # vecter

        # symbolic batch cost
        # Mean erro of observations indexed by the second last subscript
        erro = err(y, z).sum(-1).mean()
        wsum = reg(vwgt)  # weight sum
        cost = erro + wsum * self.lmd
        self.__cost__ = cost

        # symbolic gradient of cost WRT parameters
        grad = T.grad(cost, pars)
        gabs = T.concatenate([T.abs_(g.flatten()) for g in grad])
        gsup = T.max(gabs)

        # trainer control
        nwep = ((self.bt + 1) * self.bsz) // self.x.shape[-2]  # new epoch?
        
        # 2) define updates after each batch training
        up = []

        # update parameters using gradiant decent, and momentum
        for p, g in zip(pars, grad):
            # initialize accumulated gradient
            # NOTE: p.eval() causes mehem!!
            h = S(np.zeros_like(p.get_value()))

            # accumulate gradient, partially historical (due to the momentum),
            # partially noval
            up.append((h, self.mmt * h + (1 - self.mmt) * g))

            # update parameters by stepping down the accumulated gradient
            up.append((p, p - self.lrt * h))

        # update batch and eqoch index
        up.append((self.bt, (self.bt + 1) * (1 - nwep)))
        up.append((self.ep, self.ep + nwep))

        # 3) the trainer functions
        # feed symbols with actual data in batches
        _ = T.arange((self.bt + 0) * self.bsz, (self.bt + 1) * self.bsz)
        bts = {x: self.x.take(_, -2, 'wrap'), z: self.z.take(_, -2, 'wrap')}
        dts = {x: self.x, z: self.z}

        # each invocation sends one batch of training examples to the network,
        # calculate total cost and tune the parameters by gradient decent.
        self.step = F([], cost, name="step", givens=bts, updates=up)

        # training error, training cost
        self.terr = F([], erro, name="terr", givens=dts)
        self.tcst = F([], cost, name="tcst", givens=dts)

        # weights, and parameters
        self.wsum = F([], wsum, name="wsum")
        self.gsup = F([], gsup, name="gsup", givens=bts)
        # * -------- done with trainer functions -------- *

        # * -------- validation functions -------- *
        # enable validation binary disruption (binary)?
        if self.vdr:
            _ = self.__trng__.binomial(self.v.shape, 1, self.vdr, dtype=FX)
            vts = {x: self.u, z: (self.v + _) % C(2.0, FX)}
        else:
            vts = {x: self.u, z: self.v}
        self.verr = F([], erro, name="verr", givens=vts)

        # * ---------- logging and recording ---------- *
        hd, rm = [], ['step', 'gavg', 'nwgt', 'berr', 'bcst']
        for k, v in self.__dict__.items():
            if k.startswith('__') or k in rm:
                continue
            if isinstance(v, type(self.lmd)) and v.ndim < 1:
                hd.append((k, v.get_value))
            if isinstance(v, type(self.bsz)) and v.ndim < 1:
                hd.append((k, v.get_value))
            if isinstance(v, type(self.step)):
                hd.append((k, v))

        self.__head__ = hd
        self.__time__ = .0

        # the first record
        self.__hist__ = [self.__rpt__()]
        self.__gsup__ = self.__hist__[0]['gsup']
        self.__einf__ = self.__hist__[0]['terr']
        self.__nnt0__ = paint(self.nnt)

        # printing format
        self.__pfmt__ = (
            '{ep:04d}.{bt:03d}: {tcst:.2f} = {terr:.2f} + {lmd:.2e}*{wsum:.1f}'
            '|{verr:.2f}, {gsup:.2e}, {lrt:.2e}')

    # -------- helper funtions -------- *
    def nsbj(self):
        return self.x.get_value().shape[-2]

    def nbat(self):
        return self.x.get_value().shape[-2] // self.bsz.get_value()

    def yhat(self, x=None, evl=True):
        """
        Predicted outcome given input {x}. By default, use the training samples
        as input.
        """
        y = self.nnt(self.x if x is None else x)
        return y.eval() if evl else y

    def __rpt__(self):
        """ report current status. """
        s = [(k, f().item()) for k, f in self.__head__]
        s.append(('time', self.__time__))
        return dict(s)

    def __onep__(self):
        """ called on new epoch. """
        # history records
        h = self.__hist__
        
        # update the learning rate and suprimum of gradient
        # if h[-1]['gsup'] < self.__gsup__:  # accelerate
        #     self.lrt.set_value(self.lrt.get_value() * self.lrt_inc.get_value())
        #     paint(self.nnt, self.__nnt0__)
        #     self.__gsup__ = h[-1]['gsup']
        # else:                   # slow down
        #     self.lrt.set_value(self.lrt.get_value() * self.lrt_dec.get_value())
        #     paint(self.__nnt0__, self.nnt)
        if h[-1]['terr'] < self.__einf__:  # accelerate
            self.lrt.set_value(self.lrt.get_value() * self.lrt_inc.get_value())
            paint(self.nnt, self.__nnt0__)
            self.__einf__ = h[-1]['terr']
        else:                   # slow down
            self.lrt.set_value(self.lrt.get_value() * self.lrt_dec.get_value())
            paint(self.__nnt0__, self.nnt)
        

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
        b0 = bt.eval().item()   # starting batch
        ei, bi = 0, 0           # counted epochs and batches

        nep = nep + nbt // self.nbat()
        nbt = nbt % self.nbat()

        while ei < nep or bi < nbt:
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

                if self.__onep__:
                    self.__onep__()

    def cout(self, ix=None):
        """
        Print out a range of training records.

        idx: slice of records to be printed, by default the last
        one is shown.
        """
        hs = self.__hist__

        if ix is None:
            hs = [self.__hist__[-1]]
        elif ix is int:
            hs = [self.__hist__[ix]]
        else:
            hs = hs[ix]

        # latest history
        for h in hs:
            print(self.__pfmt__.format(**h))
        sys.stdout.flush()

    def query(self, fc=None, rc=None, out=None):
        """ report the trainer's history.
        fc: the field checker, can be a function taking a string field name,
        or a list of field names. by default no checking is imposed.
        rc: the record checker, a function taking a dictionary record.
        by default all record will pass.
        e.g.:
        >>> query(rc=lambda r: r['eq']<10)  # return the first 10 epoch
        
        out: the file to flush the output. if specified, the query result is
        written to a a tab-delimited file. if 0 is specified, the STDOUT will
        be used.
        """
        hs = self.__hist__
        h0 = hs[0]

        # field checker
        if fc is None:

            def fc1(f):
                return True
        elif '__iter__' in dir(fc):

            def fc1(f):
                return f in fc
        else:
            fc1 = fc

        # record checker
        if rc is None:

            def rc1(**r):
                return True
        else:
            rc1 = rc

        # numpy types from python native types
        tp = {float: 'f4', int: 'i4'}
        tp = np.dtype([(k, tp[type(v)]) for k, v in h0.items() if fc1(k)])

        # numpy data
        dt = [tuple(v for k, v in _.items() if fc1(k)) for _ in hs if rc1(**_)]
        dt = np.array(dt, tp)

        if out is not None:
            # output format
            fm = {float: '%f', int: '%d'}
            fm = '\t'.join([fm[type(v)] for k, v in h0.items() if fc1(k)])
            hd = '\t'.join(dt.dtype.names)
            np.savetxt(out, dt, fm, header=hd)

        return dt

    def reset(self):
        """ reset training records. """
        self.ep.set_value(0)
        self.bt.set_value(0)
        self.__hist__ = [self.__rpt__()]

    def plot(self, dx=None, dy=None):
        """ plot the training graph.
        dx: dimension on x axis, by default it is time.
        dy: dimensions on y axis, by defaut it is training and validation
        errors (if available).
        """
        if dx is None:
            dx = ['time']
        if dy is None:
            dy = ['terr']
            if self.u and self.v:
                dy = dy + ['verr']
        else:
            if '__iter__' not in dir(dy):
                dy = [dy]

        clr = 'bgrcmykw'
        dat = self.query(fc=dx + dy)

        plt.close()
        x = dx[0]
        for i, y in enumerate(dy):
            plt.plot(dat[x], dat[y], c=clr[i])

        return plt
