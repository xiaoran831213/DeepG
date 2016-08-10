import numpy as np

from . import hlp
from .hlp import S, T, F
import pdb


def dist_ce(y, z):
    """ symbolic expression of cross entrophy
    y: produced output
    z: expected output

    The first dimension denote unit of sampling
    """
    d = -z * T.log(y) - (1 - z) * T.log(1 - y)
    bySample = T.sum(d.flatten(ndim=2), axis=1)
    return T.mean(bySample)


def dist_l2(y, z):
    """ symbolic expression of L2 norm
    y: produced output
    z: expected output

    The first dimension denote unit of sampling
    """
    d = (y - z)**2
    bySample = T.sqrt(T.sum(d.flatten(ndim=2), axis=1))
    return T.mean(bySample)


class wreg_l2:
    """
    a callable class, returns symbolic expression of weight
    regulator in the form of L2 norm.
    """

    def __init__(self, lmb=None):
        """ the constructor.
        lmb: the relative importance of weight regulator.
        usually denoted as 'lambda' in literatures.
        """
        lmb = 0.1 if lmb is None else float(lmb)
        self.lmb = hlp.S(lmb)

    def __call__(self, lsW):
        """
        return the symbolic regulator term of a list of
        weights
        """
        w = T.concatenate([w.flatten() for w in lsW])
        return self.lmb * T.sqrt(T.sum(w**2))

    def __repr__(self):
        return 'f={}, l={}'.format('L2', self.lmb)


class Trainer(object):
    """
    Class for neural network training.
    """

    def __init__(self,
                 nnt,
                 src=None,
                 dst=None,
                 call_dist=None,
                 call_wreg=None,
                 bsz=None,
                 mmt=None,
                 lrt=None):
        """
        Constructor.
        : -------- parameters -------- :
        nnt: the neural network to be trained, could be a Nnt object so the default
        nnt.y(x) expression is used, or an symbolic expression builder function.

        src: training source, the first dimension of which stands for sample units.
        if unspecified, the trainer will try to evaluate the entry point and cache
        the result as source data.
        
        dst: training expect, the first dimension should be identical with source.
        if unspecified, a self-supervied training is assumed and the expect is set
        to be identical to the source.

        call_dist: an expression builder for ouput-expect distance(or distortion).
        the supplied builder should return an expression given the symbolic output
        {y}, and the expect {z}. the expression must evaluate to a scalar.

        call_wreg: an expression builder for regulator term.
        The build should return an expression given the list of independant weights
        to be used to panalize. the expression must evaluate to a scalar
        for the regulator. restriction oevr the magnitude of weights supposingly
        helps the final modle more generalizable.

        bsz: size of training batch
        mmt: momentom
        lrt: basic learning rate
        lmb: weight decay speed (usually denoted with 'lambda' in literatures)
        """
        # form of loss function
        self.call_dist = dist_ce if call_dist is None else call_dist

        # form of weight regulator
        self.call_wreg = (
            lambda w: S(0.0, 'lmb')) if call_wreg is None else call_wreg

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
        self.mmt = S(mmt, 'mmt')

        # learning lrt
        lrt = 0.01 if lrt is None else lrt
        self.lrt = S(lrt, 'lrt')

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
        parm = hlp.parms(y)
        npar = T.sum([p.size for p in parm])  # parameter count

        # list of independant symbolic weights to apply decay
        lswt = [p for p in parm if p.name == 'w']

        # symbolic batch cost
        dist = self.call_dist(y, z)
        wreg = self.call_wreg(lswt)
        cost = dist + wreg

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
            h = S(np.zeros_like(p.get_value()))  # p.eval() cause mehem!!

            # accumulate gradient, partially historical (due to the momentum),
            # partially noval
            up.append((h, self.mmt * h + (1 - self.mmt) * g))

            # update parameters by stepping down the accumulated gradient
            up.append((p, p - self.lrt * h))

        # update batch and eqoch index
        bat_next = (((self.bat + 1) * self.bsz) % self.src.shape[0]) // self.bsz
        eph_next = self.eph + ((self.bat + 1) * self.bsz) // self.src.shape[0]
        up.append((self.bat, bat_next))
        up.append((self.eph, eph_next))

        # from theano.ifelse import ifelse
        # gsum_prev = S(0.0)      # previous sum gradient
        # lrt_next = ifelse(T.lt(gsum, gsum_prev), self.lrt * 1.1, self.lrt / 1.2)

        # up.append((self.lrt, lrt_next))
        # up.append((gsum_prev, gsum))

        # 3) the trainer functions
        # feed symbols with explicit data in batches
        gvn = {
            x: self.src[self.bat * self.bsz:(self.bat + 1) * self.bsz],
            z: self.dst[self.bat * self.bsz:(self.bat + 1) * self.bsz]}

        # each invocation sends one batch of training examples to the network,
        # calculate total cost and tune the parameters by gradient decent.
        self.step = F([], cost, name="step", givens=gvn, updates=up)

        # return intermidiate result for batch training
        self.dist = F([], dist, name="dist", givens=gvn)
        self.wreg = F([], wreg, name="wreg")
        self.edst = F([], dist, name="edst", givens={x: self.src, z: self.dst})
        self.ecst = F([], cost, name="ecst", givens={x: self.src, z: self.dst})
        self.cost = F([], cost, name="cost", givens=gvn)
        self.grad = dict([(p, F([], g, givens=gvn)) for p, g in ZPG])
        self.npar = F([], npar, name="npar")
        self.gsum = F([], gsum, name="gsum", givens=gvn)
        self.gavg = F([], gavg, name="gavg", givens=gvn)
        # * -------- done with trainer functions -------- *

    def tune(self, nep=1, npt=1):
        """ tune the parameters by running the trainer {nep} epoch.
        an epoch is one going through of all samples
        """
        b0 = self.bat.get_value()  # starting batch
        e0 = self.eph.get_value()  # starting epoch
        eN = e0 + nep  # ending epoch
        pN = e0 + npt  # printing epoch
        pF = 'e{i:04d}: {c:08.3f} {g:07.4f}'  # printing format
        while self.eph.get_value() < eN or self.bat.get_value() < b0:
            self.step()
            i = self.eph.get_value().item()  # epoch index
            j = self.bat.get_value().item()  # batch index
            if i < pN or j < b0:
                continue
            print(pF.format(i=i, c=self.ecst().item(), g=self.gsum().item()))
            pN = i + npt  # update print epoch


def test_trainer():
    import hlp
    hlp.set_seed(120)

    import os.path as pt
    x = np.load(pt.expandvars('$AZ_SP1/lh001F1.npz'))['vtx']['tck']
    x = np.asarray(x, dtype='<f4')
    x = hlp.rescale01(x)
    d = x.shape[1]

    import sae
    from sae import SAE
    m = SAE.from_dim([d / 1, d / 2, d / 4])
    # t = Trainer(m.z, src = x, xpt = x, lrt = 0.01)
    return x, m


if __name__ == '__main__':
    pass
