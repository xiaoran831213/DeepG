import numpy as np
from theano import tensor as T
from theano import function as F
from .hlp import S, parms


def CE(y, z):
    """ symbolic expression of cross entrophy
    y: produced output
    z: expected output

    The first dimension denote unit of sampling
    """
    d = -z * T.log(y) - (1 - z) * T.log(1 - y)
    bySample = T.sum(d.flatten(ndim=2), axis=1)
    return T.mean(bySample)


def L2(y, z):
    """ symbolic expression of L2 norm
    y: produced output
    z: expected output

    The first dimension denote unit of sampling
    """
    d = (y - z)**2
    bySample = T.sqrt(T.sum(d.flatten(ndim=2), axis=1))
    return T.mean(bySample)


def L1(y, z):
    """ symbolic expression of L1 norm
    y: produced output
    z: expected output

    The first dimension of y, z denote the batch size.
    """
    d = abs(y - z)
    bySample = T.sqrt(T.sum(d.flatten(ndim=2), axis=1))
    return T.mean(bySample)


def R2(lsW):
    """
    build expression of L2 regulator given a list of weights.
    """
    w = T.concatenate([w.flatten() for w in lsW])
    return T.sqrt(T.sum(w**2))


def R1(lsW):
    """
    build expression of L1 regulator given a list of weights.
    """
    w = T.concatenate([w.flatten() for w in lsW])
    return T.sum(T.abs_(w))


def RN(lsW):
    return S(.0, name='RN')


class Trainer(object):
    """
    Class for neural network training.
    """

    def __init__(self,
                 nnt,
                 src=None,
                 dst=None,
                 err=None,
                 reg=None,
                 bsz=None,
                 mmt=None,
                 lrt=None,
                 lmd=None,
                 thd=1e-6):
        """
        Constructor.
        : -------- parameters -------- :
        nnt: an expression builder for the neural network to be trained,
        could be a Nnt object.

        src: the inputs, with the first dimension standing for sample units.
        If unspecified, the trainer will try to evaluate the entry point and
        cache the result as source data.
        
        dst: the labels, with the first dimension standing for sample units.
        if unspecified, a simi-unsupervied training is assumed as the labels
        will be identical to the inputs.

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
        # expression builder of loss
        err = CE if err is None else err

        # expression builder of weight regulator
        reg = RN if reg is None else reg

        # current epoch index
        self.eph = S(0, 'eph')

        # training batch ppsize
        bsz = 20 if bsz is None else bsz
        self.bsz = S(bsz, 'bsz')

        # current batch index
        self.bat = S(0, 'bat')

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
        self.dim = (nnt.dim[0], nnt.dim[-1])
        src = np.zeros(
            (self.bsz.get_value(), self.dim[0])) if src is None else src
        dst = np.zeros(
            (self.bsz.get_value(), self.dim[1])) if dst is None else dst
        self.src = S(src, 'src')
        self.dst = S(dst, 'dst')

        # -------- construct trainer function -------- *
        # 1) symbolic expressions
        x = T.matrix('x')  # the symbolic batch source
        z = T.matrix('z')  # the symbolic batch expect
        y = nnt(x)  # the symbolic batch output

        # list of independant symbolic parameters to be tuned
        parm = parms(y)
        npar = T.sum([p.size for p in parm])  # parameter count

        # list of  symbolic weights to apply decay
        lswt = [p for p in parm if p.name == 'w']

        # symbolic batch cost
        loss = err(y, z)  # loss function
        wsum = reg(lswt)  # weight sum
        cost = loss + wsum * self.lmd

        # symbolic gradient of cost WRT parameters
        grad = T.grad(cost, parm)
        gsum = T.sqrt(T.sum([T.square(g).sum() for g in grad]))
        gavg = gsum / npar  # average over paramter

        # 2) define updates after each batch training
        ZPG = list(zip(parm, grad))
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
        uBat = (((self.bat + 1) * self.bsz) % self.src.shape[0]) // self.bsz
        uEph = self.eph + ((self.bat + 1) * self.bsz) // self.src.shape[0]
        up.append((self.bat, uBat))
        up.append((self.eph, uEph))

        # 3) the trainer functions
        # feed symbols with explicit data in batches
        bts = {x: self.src[self.bat * self.bsz:(self.bat + 1) * self.bsz],
               z: self.dst[self.bat * self.bsz:(self.bat + 1) * self.bsz]}
        dts = {x: self.src, z: self.dst}

        # each invocation sends one batch of training examples to the network,
        # calculate total cost and tune the parameters by gradient decent.
        self.step = F([], cost, name="step", givens=bts, updates=up)

        # batch error, batch cost
        self.loss = F([], loss, name="loss", givens=bts)
        self.wsum = F([], wsum, name="wsum")
        self.cost = F([], cost, name="cost", givens=bts)

        # total error, total cost
        self.terr = F([], loss, name="terr", givens=dts)
        self.tcst = F([], cost, name="tcst", givens=dts)

        self.grad = dict([(p, F([], g, givens=bts)) for p, g in ZPG])
        self.npar = F([], npar, name="npar")
        self.gsum = F([], gsum, name="gsum", givens=bts)
        self.gavg = F([], gavg, name="gavg", givens=bts)
        # * -------- done with trainer functions -------- *

        # * -------- historical records -------- *
        self.__history__ = [self.__report__()]

    def __report__(self):
        """ report current status """
        rpt = dict(
            eph=self.eph.get_value().item(),
            bat=self.bat.get_value().item(),
            lrt=self.lrt.get_value().item(),
            lmd=self.lmd.get_value().item(),
            loss=self.loss().item(),
            wsum=self.wsum().item(),
            cost=self.cost().item(),
            gsum=self.gsum().item())
        return rpt

    def tune(self, nep=1, npt=1):
        """ tune the parameters by running the trainer {nep} epoch.
        nep: number of epoches to go through
        npt: number of epoches to hold printing
        """
        b0 = self.bat.get_value()  # starting batch
        e0 = self.eph.get_value()  # starting epoch
        eN = e0 + nep  # ending epoch
        pN = e0 + npt  # printing epoch

        while self.eph.get_value() < eN or self.bat.get_value() < b0:
            self.step()

            # should we print?
            i = self.eph.get_value().item()  # epoch index
            j = self.bat.get_value().item()  # batch index
            if i < pN or j < b0:
                continue
            self.cout()
            pN = i + npt  # update print epoch

    def cout(self):
        # printing format
        st = '{i:04d}: {c:08.3f} = {e:08.3f} + {l:4.3f} * {r:08.2f}'
        st += ', {g:07.4f}'
        st = st.format(
            i=self.eph.get_value().item(),  # epoch
            c=self.tcst().item(),           # data cost
            e=self.terr().item(),           # data error
            l=self.lmd.get_value().item(),  # lambda
            r=self.wsum().item(),           # sum of weights
            g=self.gsum().item())           # gradient
        print(st)


def test_trainer():
    """ test_trainer """
    import os.path as pt
    x = np.load(pt.expandvars('$AZ_SP1/lh001F1.npz'))['vtx']['tck']
    x = np.asarray(x, dtype='<f4')
    # x = hlp.rescale01(x)
    d = x.shape[1]

    from sae import SAE
    m = SAE.from_dim([d / 1, d / 2, d / 4])
    # t = Trainer(m.z, src = x, xpt = x, lrt = 0.01)
    return x, m
