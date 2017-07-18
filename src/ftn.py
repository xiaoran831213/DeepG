# fine-tuner for neural networks
import os
import os.path as pt
import numpy as np
from xnnt.sae import SAE
from xutl import lpz, spz


# r=main("sim/GMX/D02/0003.pgz", nep=200, wdp=2, lrt=5e-4, acc=1.01, ovr=1)
def main(fnm, **kwd):
    """ the fine-tune procedure for Stacked Autoencoder(SAE).

    -- fnm: pathname to the input, supposingly the saved progress after the
    pre-training. If {fnm} points to a directory, a file is randomly chosen
    from it.

    """
    # randomly pick pre-trained progress if {fnm} is a directory and no record
    # exists in the saved progress:
    if pt.isdir(fnm):
        fnm = pt.join(fnm, np.random.choice(os.listdir(fnm)))
    kwd.update(fnm=fnm)
    fdr, fbn = pt.split(fnm)
    fpx = fbn.split('.', 2)[0]

    # read {fnm}, but parameters in {kwd} takes precedence over those in {fnm}
    keys = list(kwd.keys()) + ['lrt']
    kwd.update((k, v) for k, v in lpz(fnm).items() if k not in keys)

    # check saved progress and overwrite options:
    sav = kwd.get('sav', '.')
    if sav is None:
        sav = pt.join(fdr, fpx)
    if pt.isdir(sav):
        sav = pt.join(sav, fpx)
    if pt.exists(sav + '.pgz') or pt.exists(sav):
        print(sav, ": exists,", )
        ovr = kwd.pop('ovr', 0)  # overwrite?
        if ovr is 0 or ovr > 2:  # do not overwrite the progress
            print(" skipped.")
            return kwd
    else:
        ovr = 2

    # resume progress, use network stored in {sav}.
    if ovr is 1:
        # remaining options in {kwd} take precedence over {sav}, but always use
        # saved network, even if there is one available in {fnm}.
        sdt = lpz(sav)          # load

        # should we restart the training due to failure?
        eot = sdt.get('eot', 0.0)
        rte = kwd.pop('rte', 1e9)
        if eot > rte:
            ovr = 2
            print('NT: RTE = {}'.format(kwd.bet('rte')))
            print('NT: EOT = {}'.format(sdt.get('eot')))
            print("NT: eot > rte, re-try.")
        else:
            kwd.pop('nwk', None)    # use saved network
            kwd.pop('lrt', None)    # use saved learning rate
            kwd.pop('eot', None)    # use saved error
            sdt.update(kwd)
            kwd = sdt
            print("continue.")
    else:                       # restart the training
        print("restart.")

    # training data, only first 500
    if 'gx0' in kwd and 'gx1' in kwd:
        gmx = np.concatenate([
            kwd['gx0'][:, np.newaxis],
            kwd['gx1'][:, np.newaxis]], 1)
    else:
        gmx = kwd['gmx']

    # take out part of the samples?
    nsb = kwd.get('nsb', None)
    if nsb is not None and nsb < gmx.shape[0]:
        idx = np.sort(np.random.permutation(gmx.shape[0])[:nsb])
        gmx = gmx[idx, :]
        kwd['sbj'] = kwd['sbj'][idx]
    else:
        nsb = gmx.shape[0]

    # genomic copy 1 & 2
    kwd['gx0'] = gmx[:, 0, :]
    kwd['gx1'] = gmx[:, 1, :]

    # training format
    xmx = gmx.reshape(nsb, -1).astype('f')

    # the dimensions
    ngv = xmx.shape[-1]
    dim = [ngv] + [ngv//2**d for d in range(1, 16) if 2**d <= ngv]

    # learing rates
    lrt = kwd.pop('lrt', .001)
    nep = kwd.pop('nep', 20)

    # Halt already?
    hte = kwd.pop('hte', .001)
    eot = kwd.pop('eot', 1e12)
    ste = kwd.pop('ste', 1e10)
    print('NT: HTE = {}'.format(hte))
    print('NT: EOT = {}'.format(eot))
    if eot < hte:               # halted?
        print('NT: eot < hte')
        print('NT: Halt.\nNT: Done.')
        return kwd
    if eot > ste:               # to much to even bother?
        print('NT: eot > ste')
        print('NT: Skip.\nNT: Done.')
        return kwd

    # train the network, create it if necessary
    nwk = kwd.pop('nwk', None)
    if nwk is None:
        nwk = SAE(dim, s='sigmoid', **kwd)
        print('create NT: ', nwk)

    # limit the working network
    wdp = kwd.pop('wdp', None)

    # wd0 indexing the lowest of new layers revealed on top of an existing
    # optimal working network
    wd0 = kwd.pop('wd0', 0)

    # pre-train the new layers when they are revealed for the first time.
    if wd0 > 0 and ovr is 2:
        # output from previously trained network at the bottom.
        xm1 = nwk.sub(None, wd0).ec(xmx).eval()
        # the new stacks.
        nw1 = nwk.sub(wd0, wdp)
        pep = kwd.pop('pep', nep)
        ptr = SAE.Train(nw1, xm1, xm1, lrt=lrt, nep=pep)
        eot = ptr.terr()
        nep = max(20, nep - pep)

    # fine-tuning
    wnk = nwk.sub(None, wdp)
    ftn = SAE.Train(wnk, xmx, xmx, lrt=lrt, hte=hte, nep=nep, **kwd)
    lrt = ftn.lrt.get_value()
    eot = ftn.terr()
    eov = ftn.verr()
    eph = kwd.pop('eph', 0) + ftn.ep.get_value()
    hof = ftn.nwk.ec(xmx).eval()
    rsd = xmx - ftn.nwk(xmx).eval()  # the residual
    if ftn.hlt:
        print('NT: Halt.')

    # 3) update progress and save.
    kwd.update(nwk=nwk, wdp=wdp, lrt=lrt,
               hof=hof, rsd=rsd,
               eot=eot, eov=eov, eph=eph)

    kwd.pop('gh0', None)
    kwd.pop('gh1', None)
    kwd.pop('gmx', None)
    if sav:
        print("write to: ", sav)
        spz(sav, kwd)
    print('NT: Done.')

    kwd = dict((k, v) for k, v in kwd.items() if v is not None)
    return kwd
