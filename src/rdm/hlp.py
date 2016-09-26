import numpy as np
import theano
import theano.tensor as T


def FX(fx=None):
    if fx is None:
        return theano.config.floatX
    else:
        theano.config.floatX = fx

# by default use 32bit float
FX('float32')

# * -------- random number helpers -------- * #
rs_np = None  # numpy random stream
rs_tn = None  # theano random stream
__seed__ = 120


def set_seed(seed):
    global rs_np, rs_tn, __seed__
    rs_np = np.random.RandomState(seed)
    rs_tn = theano.tensor.shared_randomstreams.RandomStreams(
        rs_np.randint(2**30))
    __seed__ = seed


set_seed(None)


def S(v, name=None, dtype=None, strict=False):
    """ create shared variable from v """
    if type(v) is T.sharedvar.TensorSharedVariable:
        return v
    # wrap python type to numpy type
    if not isinstance(v, np.ndarray):
        v = np.array(v, dtype)

    # wrap numeric type to default theano configuration
    if v.dtype == np.dtype('f8') and FX() is 'float32':
        v = np.asarray(v, dtype='f4')

    # if v.dtype is np.dtype('i8') and FX() is 'float32':
    #     v = np.asarray(v, dtype = 'i4')

    # if v.dtype is np.dtype('u8') and FX() is 'float32':
    #     v = np.asarray(v, dtype = 'u4')

    # broadcasting pattern
    b = tuple(s == 1 for s in v.shape)

    return theano.shared(v, name=name, strict=strict, broadcastable=b)


def shared_acc(shared_var, doc=None):
    """ build getter and setter for a shared variable """

    def acc(v=None):
        if v is None:  # getter
            return shared_var.get_value()
        else:  # setter
            # wrap python type to numpy type
            if not isinstance(v, np.ndarray):
                v = np.array(v)

            # wrap numeric type to default theano configuration
            if v.dtype is np.dtype('f8') and FX() is 'float32':
                v = np.asarray(v, dtype='f4')
            if v.dtype is np.dtype('i8') and FX() is 'float32':
                v = np.asarray(v, dtype='i4')
            if v.dtype is np.dtype('u8') and FX() is 'float32':
                v = np.asarray(v, dtype='u4')
            shared_var.set_value(v)

    # return the wrapped getter/setter
    acc.__doc__ = 'access {}.'.format(repr(shared_var)) if doc is None else doc
    return acc


# rescaled values to [0, 1]
def rescale01(x, axis=None):
    """ rescale to [0, 1] """
    return (x - x.min(axis)) / (x.max(axis) - x.min(axis))


# type checkers
def is_tvar(x):
    """ is x a theano symbolic variable with no explict value. """
    return type(x) is T.TensorVariable


def is_tshr(x):
    """ is x a theano shared variable """
    return type(x) is T.sharedvar.TensorSharedVariable


def is_tcns(x):
    """ is x a theano tensor constant """
    return type(x) is T.TensorConstant


def is_tnsr(x):
    """ is x a theano tensor """
    return (is_tvar(x) or is_tshr(x) or is_tcns(x))


# fetch parameters
def parms(y, chk=None):
    """
    find parameters in symbolic expression {y}.

    chk: checker for allagible parameter. By default only shared
    variables could pass.
    """
    chk = is_tshr if chk is None else chk

    from collections import OrderedDict

    d = OrderedDict()
    q = [y]
    while len(q) > 0:
        v = q.pop()
        q.extend(v.get_parents())
        if chk(v):
            d[v] = v

    return list(d.keys())

# def save_pgz(fo, s):
#     """ save python object to gziped pickle """
#     import gzip
#     import cPickle
#     with gzip.open(fo, 'wb') as gz:
#         cPickle.dump(s, gz, cPickle.HIGHEST_PROTOCOL)

# def load_pgz(fi):
#     """ load python object from gziped pickle """
#     import gzip
#     import cPickle
#     with gzip.open(fi, 'rb') as gz:
#         return cPickle.load(gz)
