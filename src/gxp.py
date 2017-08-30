import os.path as pt
import numpy as np
from xnnt.sae import SAE
from xutl import lpz, spz
from sklearn.decomposition import PCA


def loadVCF(vcf):
    """ Read genomic VCF file by name.
    Fix MAF and allele orders.
    return genomic matrix and subject IDs.
    """
    # Python VCF reader: pip install pyvcf
    from vcf import Reader as vcfR

    # the two homogenuous chromosomes
    A, B = [], []

    reader = vcfR(filename=vcf)
    sbj = reader.samples        # subject IDs
    for v in reader:
        # copy #1 and #2
        a = [int(g.gt_alleles[0] > '0') for g in v.samples]
        b = [int(g.gt_alleles[1] > '0') for g in v.samples]
        A.append(a)
        B.append(b)

    # compile genomic matrix
    gmx = np.array([A, B], dtype='uint8')

    # MAF fixing
    i = np.where(gmx.sum((0, 2)) > gmx.shape[2])[0]
    gmx[:, i, :] = 1 - gmx[:, i, :]

    # Allele order fix, make sure copy(a) >= copy(b)
    i = np.where(gmx[0, :, :] < gmx[1, :, :])
    gmx[0, :, :][i] = 1
    gmx[1, :, :][i] = 0

    # dim_0: sample index; dim_1: copy number; dim_2: variant index
    gmx = gmx.transpose(2, 0, 1)
    return dict(gmx=gmx, sbj=sbj)


def xpt(sav, kwd):
    """ export the progress in texture format. """
    stm = pt.join(pt.dirname(sav), pt.basename(sav).split('.')[0])
    inf = []
    for k, v in kwd.items():
        if type(v) in [int, float, str]:
            inf.append('{}={}\n'.format(k, v))
        elif isinstance(v, np.number):
            inf.append('{}={}\n'.format(k, v))
        elif isinstance(v, np.ndarray) and v.size < 2:
            inf.append('{}={}\n'.format(k, v))
        elif isinstance(v, np.ndarray) and v.ndim < 3:
            np.savetxt(stm + '.' + k, v, '%s', '\t')
        else:
            print(k, type(v), 'not exported')

    with open(stm + '.' + 'inf', 'w') as f:
        f.writelines(inf)
    return kwd


# def main(f='../raw/GXP/01/000034610', **kwd):
def main(f='../raw/GXP/01/161511549', **kwd):
    """ adaptive training for one autoendocer.
    ** ret_gmx: return only the genomic matrix, do no training.
    ** min_maf: minimum MAF filter
    ** max_maf: maximum MAF filter
    """
    # basename with directory
    idr = pt.dirname(f)                # input directory
    bsn = pt.basename(f).split('.')[0]  # basename, no surfix

    # load genotype, cache it on the first encounter.
    npz = pt.join(idr, bsn + '.npz')

    vcf = pt.join(idr, bsn + '.vcf')
    if not pt.exists(vcf):
        vcf = vcf + '.gz'

    recache = kwd.get('recache', False)
    if not pt.exists(npz) or recache:
        dat = loadVCF(vcf)
        gmx, sbj = dat['gmx'], dat['sbj']

        # MAF filtering
        maf = gmx.mean((0, 1))
        af0 = kwd.get('min_maf', .0)
        af1 = kwd.get('max_maf', .5)
        gmx = gmx[:, :, (af0 < maf) * (maf <= af1)]
        
        np.savez_compressed(npz, gmx=gmx, sbj=sbj)
    else:
        dat = np.load(npz)
        gmx, sbj = dat['gmx'], dat['sbj']
    kwd['sbj'] = sbj            # subject ID should be saved.
    kwd['gx0'] = gmx[:, 0, :]   # genomic copy 1
    kwd['gx1'] = gmx[:, 1, :]   # genomic copy 2

    # check saved progress and overwrite options:
    sav = kwd.get('sav', '.')
    if pt.isdir(sav):
        sav = pt.join(sav, bsn)
    else:
        sav = pt.join(pt.dirname(sav), pt.basename(sav).split('.')[0])
    sav = sav + '.pgz'
    if pt.exists(sav):
        print(sav, ": exists,", )
        
        # do not continue to training?
        ovr = kwd.pop('ovr', 0)
        if ovr == 0:
            print(" skipped.")
            if kwd.get('xpt', False):
                xpt(sav, lpz(sav))
            return kwd
    else:
        ovr = 2

    # resume progress, use network stored in {sav}.
    if ovr is 1:
        kwd.pop('lrt', None)  # use saved learning rate

        # remaining options in {kwd} take precedence over {sav}.
        sdt = lpz(sav)
        sdt.update(kwd)
        kwd = sdt
        print("continue training.")
    else:  # restart the training
        print("restart training.")

    nsb = gmx.shape[0]                     # sample size
    xmx = gmx.reshape(nsb, -1).astype('f')  # training data
    ngv = xmx.shape[-1]                     # feature size
    wdp = kwd.pop('wdp', 6)                 # maximum network depth
    lrt = kwd.pop('lrt', 1e-3)              # learing rates
    gdy = kwd.pop('gdy', 0)                 # greedy pre-train

    # the dimensions
    dim = [ngv] + [512//2**_ for _ in range(16) if 2**_ <= 512]
    dim = dim[:wdp]             # go at most {wdp} layers

    # perform PCA
    pca = kwd.pop('pca', True)
    if(pca):
        try:
            pca = PCA(n_components=min(xmx.shape))
            pcs = pca.fit_transform(xmx)
            pcs = pcs[:, 0:dim[-1]]
        except Exception as e:
            pcs = str(e)
        kwd.update(pcs=pcs)

    hlt = kwd.get('hlt', 0)                # halted?
    if hlt > 0:
        print('NT: Halt.\nNT: Done.')
        if kwd.get('xpt', False):
            xpt(sav, kwd)
        return kwd

    # train the network, create it if necessary
    nwk = kwd.pop('nwk', None)
    if nwk is None:
        nwk = SAE.from_dim(dim, s='sigmoid', **kwd)
        print('create NT: ', nwk)

    print('NT: begin')
    tnr = SAE.Train(nwk, xmx, xmx, lrt=lrt, gdy=gdy, **kwd)
    lrt = tnr.lrt.get_value()  # updated learning rate
    hof = nwk.ec(xmx).eval()   # high order features
    eot = tnr.terr()           # error of training
    eov = tnr.verr()           # error of validation (testing)
    hlt = tnr.hlt              # halting status

    for i in range(1, len(nwk.sa) + 1):
        ni = nwk.sub(0, i)
        ki = 'hf{}'.format(i)
        kwd[ki] = ni.ec(xmx).eval()

    # update
    kwd.update(nwk=nwk, lrt=lrt, hof=hof, eot=eot, eov=eov, hlt=hlt)

    # save the progress, then return
    spz(sav, kwd)
    if kwd.get('xpt', False):
        xpt(sav, kwd)

    if hlt > 0:
        print('NT: Halt.')
    print('NT: Done.')
    return kwd


def test():
    f = '../raw/GXP/03/002140549'
    s = '../dat'
    return main(f, sav=s, gmx=0)
