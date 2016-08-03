import numpy as np
import vcf
from pyfasta import Fasta
from random import randint
from random import choice

from dfs import CKY, NCB

from pdb import set_trace


def exp_allele(ref, alt):
    """
    Expand all sequences given the refence allele and
    a list of alternative allels
    """
    ref = str(ref)
    ALE = [ref]

    if '__iter__' in dir(alt):
        if type(alt) is str:
            alt = [alt]
        else:
            alt = [str(a) for a in alt]
    else:
        alt = [str(alt)]

    for x in alt:
        if x.startswith('<CN'):
            x = ref * int(x[3 - len(x):-1])
        ALE.append(x)

    # align alleles by filling deletion sites with '-'
    lng = max([len(a) for a in ALE])
    ALE = [a.ljust(lng, '-') for a in ALE]

    return ALE


class VcfGvr:
    """
    Genomic variant from VCF record.
    """

    def __init__(self, vcr):
        """
        vcr: variant call record.
        """
        # alleles
        ale = exp_allele(vcr.REF, vcr.ALT)

        # align alleles by filling deletion sites with '-'
        lng = len(ale[0])

        # plug in the individual genotype
        GNO = [(int(g.gt_alleles[0]), int(g.gt_alleles[1]))
               for g in vcr.samples]

        # information from the VCF record
        self.__chr__ = vcr.CHROM  # the chromosome
        self.__pos__ = vcr.start  # position in the chromosome, zero based
        self.__ref__ = vcr.REF  # the reference allele

        # derived information
        self.__ale__ = ale  # alleles
        self.__gno__ = GNO  # genotypes
        self.__idx__ = 0  # allele char index
        self.__len__ = lng  # allele string length

    def __next__(self):
        """
        return the next allele character for all subjects
        """
        if self.__idx__ == self.__len__:
            raise StopIteration()

        al = [a[self.__idx__] for a in self.__ale__]
        gt = ''.join(NCB[al[g[0]], al[g[1]]] for g in self.__gno__)

        self.__idx__ += 1
        return gt

    def rewind(self):
        self.__idx__ = 0


class VcfSeq:
    """
    A iterator class to recover FASTQ sequences of all subjects
    from a VCF file  and its genotyping reference genome.
    """

    def __init__(
            self, vgz, fsq,
            chm=None, bp0=None, bp1=None, sbj_ix=None, sbj_id=None):
        """
        vgz: bgzipped VCF file, with tabix index.
        fsq: FASTQ file storing the reference genome
        chm: the chromosome
        bp0: the starting basepair, 0 based, inclusive
        bp1: the ending basepair, 0 based, exclusive
        """
        # record the parameters
        self.vgz = vgz
        self.fsq = fsq

        # link to the *.vcf.gz
        vgz = vcf.Reader(filename=vgz)

        # link to the *.fastq
        fsq = Fasta(fsq, key_fn=lambda x: x.split()[0])

        # the chromosome
        chm = CKY[0] if chm is None else CKY[chm]
        fsq = fsq[chm]

        # starting position
        bp0 = 0 if bp0 is None else bp0

        # ending position
        if bp1 is None:
            bp1 = len(fsq)
        else:
            bp1 = min(len(fsq, bp1))

        # restrict VCF range
        vgz = vgz.fetch(chm, bp0, bp1)

        # sample size
        self.__ssz__ = len(vgz.samples)

        # the current position, 0 based
        self.__pos__ = bp0 - 1

        # the first locus in the VCF file
        try:
            while True:
                gvr = next(vgz)
                if gvr.start < bp0:
                    continue
                break
        except StopIteration:
            gvr = None
        self.__gvr__ = VcfGvr(gvr)

        # chromosome refseq, 0 based
        self.__vcf__ = vgz
        self.__chr__ = chm
        self.__bp0__ = bp0
        self.__bp1__ = bp1
        self.__fsq__ = fsq

    def __next__(self):
        """
        Iterator core method.
        """
        # advance the reference pointer if we are not going through a variant
        if not self.__gvr__ or not self.__pos__ == self.__gvr__.__pos__:
            self.__pos__ += 1

        # anounce the end of the reference sequence
        if self.__pos__ >= self.__bp1__:
            raise StopIteration()

        # are we going through a variant?
        while self.__gvr__ and self.__pos__ == self.__gvr__.__pos__:
            # the next nucleotide is in the variant
            try:
                return next(self.__gvr__)
            except StopIteration:
                self.__pos__ = self.__pos__ + len(self.__gvr__.__ref__)
                self.__gvr__ = None

            # advance the VCF to the next variant
            try:
                self.__gvr__ = VcfGvr(next(self.__vcf__))
            except StopIteration:
                break

        # not reaching the end, return the nucleotide.
        return self.__fsq__[self.__pos__] * self.__ssz__


def get_seq(vgz, fsq, chm=None, bp0=None, nbp=None):
    """
    Restore part of the sequence from a reference sequence
    (FASTQ) and a variant file (VCF).
    vgz: name of the bgzipped VCF file (*.vcf.gz).
    fsq: the reference sequence file in FASTQ format.
    chm: chromosome to be sampled, zero based.
    bp0: initial position in the chromosome, zero based.
    nbp: number of basepairs to sample, (defaut=1024).
    """
    if nbp is None:
        nbp = 1024

    itr = VcfSeq(vgz, fsq, chm=chm, bp0=bp0)
    
    # ret = np.empty((nbp, itr.__ssz__), dtype='u1')
    ret = []
    for i in range(nbp):
        ret.append(next(itr))

    return ret


class RndVsq:
    """
    Randomly draw some variable sequence from a reference
    sequence (FASTQ) and a variant file (VCF).
    """
    def __init__(self, vgz, fsq,
                 chm=None, bp0=0, bp1=None, wnd=1024, ncb=str):
        """
        vgz: name of the bgzipped VCF file (*.vcf.gz), if None,
        only the reference genome will be returned.

        fsq: the reference sequence file in FASTQ format, it must
        not be None.

        chm: chromosome to be sampled, zero based.
        bp0: initial position in the chromosome, zero based.
        wnd: sample window size, (defaut=1024).

        ncb: nucleotide coding base, can be str or int.
        """
        self.VGZ = vgz
        self.FSQ = fsq

        # link to the *.vcf.gz
        vgz = vcf.Reader(filename=vgz)

        # link to the *.fastq
        fsq = Fasta(fsq, key_fn=lambda x: x.split()[0])

        # get the first variant
        gvr = next(vgz)

        # the chromosome
        chm = gvr.CHROM if chm is None else CKY[chm]
        fsq = fsq[chm]
        self.CHM = chm

        # starting and ending positions
        bp1 = len(fsq) if bp1 is None else min(len(fsq), bp1)
        self.BP0 = bp0
        self.BP1 = bp1

        # sanity check
        if bp1 - bp0 < 1024:
            raise Exception('bp1 - bp0 < 1024!')

        # nucleited encoding
        self.NCB = ncb
        ncb = NCB[ncb]

        # private members
        self.__vgz__ = vgz
        self.__fsq__ = fsq
        self.__gvr__ = gvr
        self.__chm__ = chm
        self.__fsq__ = fsq
        self.__bp0__ = bp0
        self.__bp1__ = bp1
        self.__wnd__ = wnd
        self.__ncb__ = ncb
        self.__mem__ = []

    def get_seq(self):
        # locate one variant first, then fetch the surrounding region.
        vz = self.__vgz__
        ch = self.__chm__
        ps = None
        while ps is None:
            ps = randint(self.__bp0__, self.__bp1__ - 1)
            try:
                ps = next(vz.fetch(ch, ps, self.__bp1__)).start
            except StopIteration:
                ps = None

        # half left & half right
        hl = int(self.__wnd__/2); hr = self.__wnd__ - hl
        # startign and ending basepair
        ps = max(ps, self.__bp0__ + hl)
        ps = min(ps, self.__bp1__ - hr)
        st, ed = ps - hl, ps + hl

        gvr = [g for g in self.__vgz__.fetch(ch, st, ed)]
        ps = st
        s0, s1 = [], []              # 2 copies of chromosome
        # randomly pick one allele for the two copies of sequences
        for v in gvr:
            # the static sequence between previous and current variant
            s0.append(self.__fsq__[ps:v.start])
            s1.append(self.__fsq__[ps:v.start])
            # the current variant
            al = exp_allele(v.REF, v.ALT)
            s0.append(choice(al))
            s1.append(choice(al))
            # advance the pointer

            # advance the reference pointer
            ps = v.start + len(v.REF)

        # rest of the sequence
        s0.append(self.__fsq__[ps:ed])
        s1.append(self.__fsq__[ps:ed])

        # unlist the sequence
        s0 = ''.join(s0)
        s1 = ''.join(s1)

        # get nucleotide coding bases
        sq = [self.__ncb__[x] for x in zip(s0, s1)]
        if self.NCB is str:
            sq = ''.join(sq)
        return sq


def test1():
    vgz = '../raw/wgs/22.vcf.gz'
    fsq = '../raw/hs37d5.fa'
    # r = VcfSeq(vgz, fsq, chm=22 - 1, bp0=16050072)
    # r = VcfSeq(vgz, fsq, chm=22 - 1, bp0=16050650)
    r = get_seq(vgz, fsq, chm=22 - 1, bp0=16050650)
    # r = get_seq(vgz, fsq, chm=22 - 1, bp0=14050650)
    return r


def test2():
    vgz = '../raw/ann/19.vcf.gz'
    fsq = '../raw/hs37d5.fa'
    ret = RndVsq(vgz, fsq, ncb=int)
    return ret
