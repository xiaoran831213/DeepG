import numpy as np
from theano import tensor as T
from theano import function as F
from .hlp import S, parms
from time import time as tm


def CE(y, z):
    """ symbolic expression of cross entrophy
    y: produced output
    z: expected output

    The first dimension denote unit of sampling
    """
    return -(z * T.log(y) + (1 - z) * T.log(1 - y))


def L2(y, z):
    """ symbolic expression of L2 norm
    y: produced output
    z: expected output

    The first dimension denote unit of sampling
    """
    return T.sqrt((y - z)**2)


def L1(y, z):
    """ symbolic expression of L1 norm
    y: produced output
    z: expected output

    The first dimension of y, z denote the batch size.
    """
    return abs(y - z)


def R0(x, thd=1e-06):
    """ build expression of L0 norm given a vector. """
    return T.sum((T.abs_(x) > thd), dtype='float32')


def R1(x):
    """ build expression of L1 norm given a vector. """
    return T.sum(T.abs_(x))


def R2(x):
    """ build expression of L2 norm given a vector. """
    return T.sqrt(T.sum(x**2))


def RN(lsW):
    return S(.0, name='RN')


class Trainer(object):
    """
    Class for neural network training.
    """

    def __init__(self,
                 nnt,
                 x=None,
                 z=None,
                 v_x=None,
                 v_z=None,
                 err=None,
                 reg=None,
                 bsz=None,
                 mmt=None,
                 lrt=None,
                 lmd=None,
                 rpt_frq=None,
                 thd=1e-6):
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

        v_x: the valication data inputs
        v_z: the validation data labels

        err: an expression builder for training error.
        the builder should return an expression given the symbolic output
        {y}, and the label {z}. the expression must evaluate to a scalar.

        reg: an expression builder for weight regulator.
        The builder should return an expression given the list of  weights
        to be shrunken. The expression must evaluate to a scalar.

        bsz: size of a training batch
        mmt: momentom
        lrt: basic learning rate
        lmb: weight decay factor, the lambda
        """
        # expression builder of error
        err = CE if err is None else err

        # expression builder of weight regulator
        reg = RN if reg is None else reg

        # current epoch index
        self.ep = S(0, 'ep')

        # training batch ppsize
        bsz = 20 if bsz is None else bsz
        self.bsz = S(bsz, 'bsz', dtype='u4')

        # current batch index
        self.bt = S(0, 'bt')

        # momentumn, make sure momentum is a sane value
        mmt = 0.0 if mmt is None else mmt
        assert mmt < 1.0 and mmt >= 0.0
        self.mmt = S(mmt, 'MMT')

        # learning rate
        lrt = 0.01 if lrt is None else lrt
        self.lrt = S(lrt, 'LRT')

        # the raio of weight decay, lambda
        lmd = 0.1 if lmd is None else lmd
        self.lmd = S(lmd, 'LMD')

        # grand source and expect
        self.nnt = nnt
        self.dim = (nnt.dim[0], nnt.dim[-1])
        x = S(np.zeros(bsz, self.dim[0]) if x is None else x)
        z = S(np.zeros(bsz, self.dim[1]) if z is None else z)
        self.x, self.z = x, z

        v_x = S(v_x) if v_x is not None else v_x
        v_z = S(v_z) if v_z is not None else v_z
        self.v_x, self.v_z = v_x, v_z

        # -------- helper expressions -------- *
        nsbj = T.cast(self.x.shape[-2], 'int32')
        nbat = nsbj // self.bsz

        # -------- construct trainer function -------- *
        # 1) symbolic expressions
        x = T.tensor(name='x', dtype=x.dtype, broadcastable=x.broadcastable)
        z = T.tensor(name='z', dtype=z.dtype, broadcastable=z.broadcastable)
        y = nnt(x)  # the symbolic batch output

        # list of independant symbolic parameters to be tuned
        pars = parms(y)  # parameters
        npar = T.sum([p.size for p in pars])  # count

        # list of  symbolic weights to apply decay
        wgts = [p for p in pars if p.name == 'w']  # weights
        vwgt = T.concatenate([w.flatten() for w in wgts])  # vecter
        nwgt = vwgt.size  # count
        wstd = T.std(vwgt)  # std.

        # symbolic batch cost
        # Mean erro of observations indexed by the second last subscript
        erro = err(y, z).sum(-1).mean()
        wsum = reg(vwgt)  # weight sum
        cost = erro + wstd * self.lmd

        # symbolic gradient of cost WRT parameters
        grad = T.grad(cost, pars)
        gsum = T.sqrt(T.sum([T.square(g).sum() for g in grad]))
        gavg = gsum / npar  # average over paramter

        # 2) define updates after each batch training
        ZPG = list(zip(pars, grad))
        up = []
        # update parameters using gradiant decent, and momentum
        for p, g in ZPG:
            # initialize accumulated gradient
            # NOTE: p.eval() causes mehem!!
            h = S(np.zeros_like(p.get_value()))

            # accumulate gradient, partially historical (due to the momentum),
            # partially noval
            up.append((h, self.mmt * h + (1 - self.mmt) * g))

            # update parameters by stepping down the accumulated gradient
            up.append((p, p - self.lrt * h))

        # update batch and eqoch index
        uBat = (((self.bt + 1) * self.bsz) % self.x.shape[-2]) // self.bsz
        uEph = self.ep + ((self.bt + 1) * self.bsz) // self.x.shape[-2]
        up.append((self.bt, uBat))
        up.append((self.ep, uEph))

        # 3) the trainer functions
        # feed symbols with explicit data in batches
        # bts = {x: self.x[self.bt * self.bsz:(self.bt + 1) * self.bsz],
        #        z: self.z[self.bt * self.bsz:(self.bt + 1) * self.bsz]}
        bts = T.arange((self.bt + 0) * self.bsz, (self.bt + 1) * self.bsz)
        bts = {x: self.x.take(bts, -2, 'wrap'),
               z: self.z.take(bts, -2, 'wrap')}
        dts = {x: self.x, z: self.z}

        # each invocation sends one batch of training examples to the network,
        # calculate total cost and tune the parameters by gradient decent.
        self.step = F([], cost, name="step", givens=bts, updates=up)

        # help functions
        self.nbat = F([], nbat, name="nbat")
        self.nsbj = F([], nsbj, name="nsbj")

        # batch error, batch cost
        self.berr = F([], erro, name="berr", givens=bts)
        self.bcst = F([], cost, name="cost", givens=bts)

        # training error, training cost
        self.terr = F([], erro, name="terr", givens=dts)
        self.tcst = F([], cost, name="tcst", givens=dts)

        # weights, and parameters
        self.wsum = F([], wsum, name="wsum")
        self.wstd = F([], wstd, name="wstd")
        self.nwgt = F([], nwgt, name="nwgt")
        self.grad = dict([(p, F([], g, givens=bts)) for p, g in ZPG])
        self.npar = F([], npar, name="npar")
        self.gsum = F([], gsum, name="gsum", givens=bts)
        self.gavg = F([], gavg, name="gavg", givens=bts)
        # * -------- done with trainer functions -------- *

        # * -------- validation functions -------- *
        if self.v_x is not None and self.v_z is not None:
            vts = {x: self.v_x, z: self.v_z}
            self.verr = F([], erro, name="verr", givens=vts)
        else:
            self.verr = lambda: 0

        # * -------- historical records -------- *
        hd, rm = [], ['step', 'gavg', 'nwgt', 'berr', 'bcst']
        for k, v in self.__dict__.items():
            if k.startswith('__') or k in rm:
                continue
            if isinstance(v, type(self.lmd)) and v.ndim < 1:
                hd.append((k, v.get_value))
            if isinstance(v, type(self.step)):
                hd.append((k, v))
        self.__head__ = hd

        self.__time__ = 0.0

        # the first record
        self.__hist__ = [self.__rpt__()]

        # printing format
        self.__pfmt__ = ('{ep:04d}.{bt:03d}: '
                         '{tcst:.2f} = {terr:.2f} + {lmd:.2e} * {wsum:.1f}, '
                         '{wstd:.2f}, {gsum:.1f}, {lrt:.2e}')

    # prediction (internal sample)
    def yhat(self, x=None):
        """ calculate predicted outcome given input {x}. By default,
        the training samples are to be sent as {x}."""
        if x is None:
            x = self.x
        return self.nnt(x)

    def __rpt__(self):
        """ report current status. """
        s = [(k, f().item()) for k, f in self.__head__]
        s.append(('time', self.__time__))

        return dict(s)

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

        nep = nep + nbt // self.nbat().item()
        nbt = nbt % self.nbat().item()

        while ei < nep or bi < nbt:
            # send one batch for training
            t0 = tm()
            self.step()
            self.__time__ += tm() - t0

            bi = bi + 1  # batch count increase by 1
            if rec > 0:  # record each batch
                self.__hist__.append(self.__rpt__())
            if prt > 0:  # print each batch
                self.cout()

            # see the next epoch relative to batch b0?
            if bt.eval().item() == b0:
                ei = ei + 1  # epoch count increase by 1
                bi = 0  # reset batch count

            # record or print after an epoch
            if bt.eval().item() == 0:
                if rec == 0:  # record at new epoch
                    self.__hist__.append(self.__rpt__())
                if prt == 0:  # print
                    self.cout()

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
            def fc1(f): return True
        elif '__iter__' in dir(fc):
            def fc1(f): return f in fc
        else:
            fc1 = fc

        # record checker
        if rc is None:
            def rc1(**r): return True
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
