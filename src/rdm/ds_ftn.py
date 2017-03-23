# fine-tuner for neural networks
import numpy as np
import os
import sys
from os import path as pt
from tnr.cmb import Comb as Tnr
from xutl import spz, lpz
sys.path.extend(['..'] if '..' not in sys.path else [])
from pdb import set_trace


def ftn_sae(w, x, u=None, nft=None,
            ae0=None, ae1=None, lrt=.01, hte=0.005, **kwd):
    """ layer-wise unsupervised pre-training for
    stacked autoencoder.
    w: the stacked autoencoders
    x: the inputs.

    nft: maximum number of epochs to go through for fine-tuning.
    ae0: which autoencoder in the stack to start the tuning?
    ae1: which autoencoder in the stack to end the tuning?

    By default, the the entire SAE of all layers are tuned, that is,
    start = 0, depth = len(w.sa)

    kwd: additional key words to pass on to the trainer.
    """
    # number of epoch to go through
    ftn = kwd.get('ftn', None)

    # validation data set
    u = x if u is None or u.size is 0 else u

    # select sub-stack
    w = w.sub(ae1, ae0) if ae0 or ae1 else w
    x = w.sub(ae0, 0).ec(x).eval() if ae0 else x
    u = w.sub(ae0, 0).ec(u).eval() if ae0 and u else u

    # build the trainer of fine-tuning:
    ftn = Tnr(w, x, u=u, lrt=lrt, hte=hte, **kwd)

    # fine-tune
    nft = 20 if nft is None else nft
    ftn.tune(nft)

    kwd.update(ftn=ftn, nft=nft, ae0=ae0, ae1=ae1)
    kwd = dict((k, v) for k, v in kwd.iteritems() if v is not None)

    return kwd


def main(fnm='../../sim/W09/DS1_PTN', **kwd):
    """ the fine-tune procedure for Stacked Autoencoder(SAE).

    -- fnm: pathname to the input, supposingly the saved progress after the
    pre-training. If {fnm} points to a directory, a file is randomly chosen
    from it.

    ** ae1: depth of the sub SA.
    """
    # randomly pick pre-trained progress if {fnm} is a directory and no record
    # exists in the saved progress:
    if pt.isdir(fnm):
        fnm = pt.join(fnm, np.random.choice(os.listdir(fnm)))
    kwd.update(fnm=fnm)

    # load data from {fnm}, but let parameters in {kwd} takes precedence over
    # those in {fnm}
    _ = kwd.keys()
    kwd.update((k, v) for k, v in lpz(fnm).iteritems() if k not in _)

    # check saved progress and overwrite options:
    sav = kwd.get('sav', pt.basename(fnm).split('.')[0])
    if pt.exists(sav + '.pgz'):
        print(sav, ": exists,", )
        ovr = kwd.pop('ovr', 0)  # overwrite?

        if ovr is 0 or ovr > 2:  # do not overwrite the progress
            print(" skipped.")
            return kwd
    else:
        ovr = 2

    # resume progress, use network stored in {sav}.
    if ovr is 1:
        kwd.pop('nwk', None)    # use saved network for training
        # kwd.pop('lrt', None)    # use saved learning rate for training

        # remaining options in {kwd} take precedence over {sav}.
        sdt = lpz(sav)
        sdt.update(kwd)
        kwd = sdt
        print("continue training.")
    else:                       # restart the training
        # kwd.pop('lrt', None)    # do not use archived NT LRT
        print("restart training.")

    # <-- __x, w, npt, ptn, ... do it.
    xmx = kwd['xmx']            # training data
    nwk = kwd['nwk']            # the whole network, not subset
    dph = len(nwk.sa)           # depth of the network
    ae1 = kwd.get('ae1', dph)   # encoding depth -- autoencoder one

    # learing rates for normal training
    lrt = kwd.pop('lrt', .01)

    # create error tables if necessary:
    if kwd.get('etb') is None:
        kwd['etb'] = np.zeros([dph + 1, 2]) - 1.0
        kwd['etb'][0] = 0       # raw data has zero error
    etb = kwd['etb']

    # create high order feature (HOF) table:
    if kwd.get('hof') is None:
        kwd['hof'] = [None] * (dph + 1)
    hof = kwd['hof']

    # raw data as trivial HOF
    _ = kwd['gmx'][:, :, kwd['ugv'] < 1]
    _ = _.reshape(_.shape[0], -1)
    hof[0] = _

    # fine-tuning
    kwd = ftn_sae(nwk, xmx, xmx, lrt=lrt, **kwd)
    ftn = kwd.pop('ftn')
    lrt = ftn.lrt.get_value()  # learning rate
    etb[ae1, 0] = ftn.terr()   # training error
    etb[ae1, 1] = ftn.verr()   # validation error
    if ftn.hlt.get_value():
        print('NT: halted.')

    # high order features for all individuals, with typed variants
    _ = kwd['gmx'][:, :, kwd['ugv'] < 1]
    _ = _.reshape(_.shape[0], -1)
    hof[ae1] = ftn.nnt.ec(_).eval()

    # 3) update progress and saving.
    kwd.update(lrt=lrt, etb=etb, hof=hof)
    if sav:
        print("write to: ", sav)
        spz(sav, kwd)

    kwd = dict((k, v) for k, v in kwd.iteritems() if v is not None)
    return kwd


def ept(fnm, out=None):
    """ Export training result in text format.
    -- fnm: filename of training progress.
    -- sav: where to save the export.
    """
    pwd = os.getcwd()
    fnm = pt.abspath(fnm)
    
    import tempfile
    tpd = tempfile.mkdtemp()
    if out is None:
        out = pwd
    if pt.isdir(out):
        out = pt.join(out, pt.basename(fnm).split('.')[0])
    if not out.endswith('.tgz') or out.endswith('.tar.gz'):
        out = out + '.tgz'
    out = pt.abspath(out)

    # read the training progress
    dat = lpz(fnm)
    [dat.pop(_) for _ in ['xmx', 'sav', 'nep', 'nft', 'ovr']]
    dat['fnm'] = fnm
    dat['out'] = out

    # genomic map
    np.savetxt(pt.join(tpd, 'gmp.txt'), dat.pop('gmp'), '%d\t%d\t%s')

    # genomic matrix
    gmx = dat.pop('gmx')
    np.savetxt(pt.join(tpd, 'gx0.txt'), gmx[:, 0, :], '%d')
    np.savetxt(pt.join(tpd, 'gx1.txt'), gmx[:, 1, :], '%d')

    # genomic data in dosage format
    np.savetxt(pt.join(tpd, 'dsg.txt'), dat.pop('dsg'), '%d')

    # subjects
    np.savetxt(pt.join(tpd, 'sbj.txt'), dat.pop('sbj'), '%s')

    # untyped subjects (indices)
    np.savetxt(pt.join(tpd, 'usb.txt'), dat.pop('usb'), '%d')

    # untyped variants (indices)
    np.savetxt(pt.join(tpd, 'ugv.txt'), dat.pop('ugv'), '%d')

    # high-order features
    hof = dat.pop('hof')
    for i in range(len(hof)):
        if hof[i] is None:
            continue
        np.savetxt(pt.join(tpd, 'hf{}.txt'.format(i)), hof[i], '%.8f')

    # error table
    np.savetxt(pt.join(tpd, 'etb.txt'), dat.pop('etb'), '%.8f')

    # neural networks
    nwk = dat.pop('nwk')
    ns = np.savetxt
    for i in range(len(nwk)/2):
        if hof[i] is None:
            continue
        ec, dc = nwk[i], nwk[-1-i]
        ns(pt.join(tpd, 'ew{}.txt'.format(i)), ec.w.eval(), '%.10f')
        ns(pt.join(tpd, 'eb{}.txt'.format(i)), ec.b.eval(), '%.10f')
        ns(pt.join(tpd, 'dw{}.txt'.format(i)), dc.w.eval(), '%.10f')
        ns(pt.join(tpd, 'db{}.txt'.format(i)), dc.b.eval(), '%.10f')

    # meta information
    inf = open(pt.join(tpd, 'inf.txt'), 'w')
    for k, v in dat.iteritems():
        inf.write('{}={}\n'.format(k, v))
    inf.close()                 # done

    # pack the output, delete invididual files
    import tarfile
    import shutil

    # packing
    os.chdir(tpd)               # goto the packing dir
    try:
        tar = tarfile.open(out, 'w:gz')
        [tar.add(_) for _ in os.listdir('.')]
        shutil.rmtree(tpd, True)
        tar.close()
    except Exception as e:
        print(e)
    os.chdir(pwd)               # back to the working dir
