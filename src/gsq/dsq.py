# generate dosage sequence from VCF
from vcf import Reader as vcfR
from random import randint
from random import random
from gsq.dfs import CKY


class DsgSeq:
    """
    A iterator class to recover FASTQ sequences of all subjects
    from a VCF file  and its genotyping reference genome.
    """

    def __init__(self, vgz, chm=None, bp0=None, bp1=None, sbj=None):
        """
        vgz: bgzipped VCF file, with tabix index.
        fsq: FASTQ file storing the reference genome
        chm: the chromosome
        bp0: the starting basepair, 0 based, inclusive
        bp1: the ending basepair, 0 based, exclusive
        """
        # record the parameters
        self.VGZ = vgz

        # find the first locus in the VCF file
        gv1 = next(vcfR(filename=vgz))

        # VGZ reader point to the *.vcf.gz
        vgz = vcfR(filename=vgz)

        # the chromosome and its length
        chm = gv1.CHROM if chm is None else CKY[chm]
        cln = vgz.contigs[chm].length

        # starting position
        if not bp0:
            bp0 = gv1.start
        elif bp0 < 0:
            bp0 = gv1.start + bp0

        # ending position
        if not bp1:
            bp1 = cln
        elif bp1 < 0:
            bp1 = cln + bp1

        # restrict VCF range
        vgz = vgz.fetch(chm, bp0, bp1)

        # find the first locus in the region
        gvr = None
        while gvr is None:
            try:
                gvr = next(vgz)
            except StopIteration:
                break
            if gvr.start < bp0:
                gvr = None

        # private members
        self.__pos__ = bp0  # the pointer
        self.__gvr__ = gvr  # the variant
        self.__vgz__ = vgz  # vcf reader
        self.__bp0__ = bp0  # starting position
        self.__bp1__ = bp1  # ending position

    def __next__(self):
        """
        Iterator core method.
        """
        # anounce the end of the reference sequence
        if not self.__pos__ < self.__bp1__:
            raise StopIteration()

        # return dosage values if the pointer is on a variant
        if self.__gvr__ and self.__pos__ == self.__gvr__.start:
            # get dosage values
            dsg = [int(g.gt_alleles[0] > '0') + int(g.gt_alleles[1] > '0')
                   for g in self.__gvr__.samples]

            # advance pointer by the size of variant
            self.__pos__ = self.__gvr__.end

            # locate next variant
            try:
                self.__gvr__ = next(self.__vgz__)
            except StopIteration:
                self.__gvr__ = None
        else:
            # return 0 for all subjects if the pointer is not on a variant
            dsg = [0] * len(self.__vgz__.samples)

            # advance the point by 1 nucleotide
            self.__pos__ += 1

        dsg = ''.join([str(g) for g in dsg])
        return dsg


class RndDsq:
    """
    Randomly draw some sequence from a variant file (VCF).
    """

    def __init__(
            self, vgz, chm=None, bp0=None, bp1=None, wnd=1024, dsg='012'):
        """
        vgz: name of the bgzipped VCF file (*.vcf.gz), if None,
        only the reference genome will be returned.

        chm: chromosome to be sampled, zero based.
        bp0: initial position in the chromosome, zero based.
        wnd: sample window size, (defaut=1024).
        dsg: encoding for alternative allele count 0, 1, and 2; use '012' for
        additive, '022' for dominative, and '002' for recessive encoding. The
        default scheme is addtive, '012'.
        """
        self.VGZ = vgz
        # find the first locus in the VCF file
        gv1 = next(vcfR(filename=vgz))

        # VGZ reader point to the *.vcf.gz
        vgz = vcfR(filename=vgz)

        # the chromosome and its length
        chm = gv1.CHROM if chm is None else CKY[chm]
        cln = vgz.contigs[chm].length
        self.CHR = chm

        # starting position
        if not bp0:
            bp0 = gv1.start
        elif bp0 < 0:
            bp0 = gv1.start + bp0
        self.BP0 = bp0

        # ending position
        if not bp1:
            bp1 = cln
        elif bp1 < 0:
            bp1 = cln + bp1
        self.BP1 = bp1

        # sanity check
        if bp1 - bp0 < 1024:
            raise Exception('bp1 - bp0 < 1024!')

        # private members
        self.__vgz__ = vgz
        self.__gvr__ = gv1
        self.__chm__ = chm
        self.__bp0__ = bp0
        self.__bp1__ = bp1
        self.__wnd__ = wnd
        self.__dsg__ = dsg
        self.__mem__ = []

    def __next__(self):
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

        # startign and ending basepair
        st = max(ps - randint(0, self.__wnd__-1), self.__bp0__)
        st = min(st, self.__bp1__ - self.__wnd__)
        ed = st + self.__wnd__

        # find variants in the sample window
        vs = [g for g in self.__vgz__.fetch(ch, st, ed) if st <= g.start]

        # initizlize pointer, and dosage value sequence
        ps, sq = st, []

        # randomly pick one allele for the two copies of sequences
        for v in vs:
            # get allele frequency
            af = v.INFO.get('AF')
            if af:
                af = sum(af)
            else:
                af = max(1, v.num_het + 2 * v.num_hom_alt) / max(
                    len(v.samples), 2)

            # generate alleles
            al = self.__dsg__[int(random() < af) + int(random() < af)]

            # a) current variant is ahead of the previous one, fill up the
            # gap in between
            if (v.start > ps):
                sq.append('0' * (v.start - ps))

            # b) current is overlapping with the previous one, indicating a
            # multi-allelic locus, and an extra chance to get ALT allele
            if (v.start < ps):
                al = self.__dsg__[max(2, int(sq.pop()) + int(al))]

            # update the nucleotide pointer
            sq.append(al)
            ps = v.start + 1

        # pad the sequence if necessary
        if ps < ed:
            sq.append('0' * (ed - ps))

        # join the sequence
        dt = ''.join(sq)
        return dt


def test1():
    v = '../raw/ann/03.vcf.gz'
    r = DsgSeq(v, chm=3 - 1, bp0=16050072)
    # r = VcfSeq(vgz, fsq, chm=22 - 1, bp0=16050650)
    return r
