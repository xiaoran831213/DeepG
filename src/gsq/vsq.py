# dosage variants sequence from VCF
from vcf import Reader as vcfR
from random import randint
from random import random
from gsq.dfs import CKY
import numpy as np


class DsgVsq:
    """
    A iterator class to fetch dosage values from a VCF file.
    """

    def __init__(self, vgz, chm=None, bp0=0, bp1=None, wnd=1024, dsg='012'):
        """
        vgz: bgzipped VCF file, with tabix index.
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

        # ending position
        if not bp1:
            bp1 = cln
        elif bp1 < 0:
            bp1 = cln + bp1

        # restrict VCF range
        vgz = vgz.fetch(chm, bp0, bp1)

        # private members
        self.__pos__ = bp0  # the pointer
        self.__vgz__ = vgz  # vcf reader
        self.__bp0__ = bp0  # starting position
        self.__bp1__ = bp1  # ending position
        self.__wnd__ = wnd
        self.__dsg__ = dsg

    def __next__(self):
        d = np.zeros((len(self.__vgz__.samples), self.__wnd__), '<i1')
        i = 0
        while (i < self.__wnd__):
            # get next variant
            try:
                v = next(self.__vgz__)
            except StopIteration as e:
                raise e

            # get dosage values
            v = [int(g.gt_alleles[0] > '0') + int(g.gt_alleles[1] > '0')
                 for g in v.samples]
            d[:, i] = np.array(v, '<i1')
            i = i + 1

        # the coding of allele counts
        d = np.array([int(c) for c in self.__dsg__], '<i1')[d]

        return d

    # compatible with Python 2.x
    def next(self):
        return self.__next__()


class RndVsq:
    """
    Randomly draw a squence of variants from a variant file (VCF), ignoring
    non-variants in between.
    """

    def __init__(self, vgz, chm=None, bp0=None, bp1=None, wnd=1024, dsg='012'):
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
        ps = randint(self.__bp0__, self.__bp1__ - 1)

        # find variants in the sample window
        vz = self.__vgz__.fetch(self.__chm__, ps)

        sq = []
        while len(sq) < self.__wnd__:
            try:
                v = next(vz)
            except StopIteration:
                break

            # get allele frequency
            af = v.INFO.get('AF')
            if af:
                af = sum(af)
            else:
                af = max(1, v.num_het + 2 * v.num_hom_alt) / max(
                    len(v.samples), 2)

            # generate alleles
            al = self.__dsg__[int(random() < af) + int(random() < af)]

            # update the nucleotide pointer
            sq.append(al)

        # pad the sequence if necessary
        if len(sq) < self.__wnd__:
            sq.append('0' * (self.__wnd__ - len(sq)))

        # join the sequence
        dt = ''.join(sq)
        return dt

    # compatible with Python 2.x
    def next(self):
        return self.__next__()


def test1():
    v = '../raw/ann/03.vcf.gz'
    r = DsgVsq(v, chm=3 - 1, bp0=16050072)
    # r = VcfSeq(vgz, fsq, chm=22 - 1, bp0=16050650)
    return r
