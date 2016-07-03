## deal with reference genome in FASTQ format
from pyfasta import Fasta
import vcf
import pdb
import os
import os.path as pt

## IUPAC nucleotide code Base
NCB = {
    ('A', 'A'): 'A',                             # Adenine
    ('C', 'C'): 'C',                             # Cytosine
    ('G', 'G'): 'G',                             # Guanine
    ('T', 'T'): 'T',                             # Thymine
    
    ('A', 'G'): 'R',
    ('C', 'T'): 'Y',
    ('G', 'C'): 'S',
    ('A', 'T'): 'W',
    ('G', 'T'): 'K',
    ('A', 'C'): 'M',

    ('G', 'A'): 'R',
    ('T', 'C'): 'Y',
    ('C', 'G'): 'S',
    ('T', 'A'): 'W',
    ('T', 'G'): 'K',
    ('C', 'A'): 'M',

    ('-', 'A'): 'B',
    ('-', 'C'): 'D',
    ('-', 'G'): 'H',
    ('-', 'T'): 'V',

    ('A', '-'): 'B',
    ('C', '-'): 'D',
    ('G', '-'): 'H',
    ('T', '-'): 'V',
    
    ('-', '-'): '-'}

class VcfGvr:
    """
    Genomic variant from VCF record.
    """
    def __init__(self, vc):
        """
        vc: variant call record.
        """
        ## alleles
        ALE  = [vc.REF]
        for x in vc.ALT:
            x = str(x)
            ## copy number vars(iation
            if x.startswith('<CN'):               
                x = vc.REF * int(x[3 - len(x):-1])
            ALE.append(x)

        ## fill deletion sites with '-' to align alleles
        SLN = max([len(x) for x in ALE])
        ALE = [x.ljust(SLN, '-') for x in ALE]

        ## plug in the individual genotype
        GNO = [(int(g.gt_alleles[0]), int(g.gt_alleles[1])) for g in vc.samples]

        self.__ale__ = ALE            # alleles
        self.__gno__ = GNO            # genotypes
        self.__len__ = SLN            # allele string length
        self.__pos__ = 0              # allele char position

    def __next__(self):
        """
        return the next allele character for all subjects
        """
        if self.__pos__ == self.__len__:
            raise StopIteration()

        al = [a[self.__pos__] for a in self.__ale__]
        gt = ''.join(NCB[al[g[0]], al[g[1]]] for g in self.__gno__)

        self.__pos__ += 1
        return gt
    
class VcfSeq:
    """
    A iterator class to recover FASTQ sequences of all subjects
    from a VCF file  and its genotyping reference genome.
    """
    def __init__(self, vgz, fsq, CHR = None, BP0 = None, NBP = None):
        """
        vgz: bgzipped VCF file, with tabix index.
        fsq: FASTQ file storing the reference genome
        CHR: the chromosome
        BP1: the initial basepair, 0 based, inclusive
        BP2: the ending basepair, 0 based, exclusive
        nbp: restrict the number of basepairs to go through
        """
        ## for queries
        self.vgz = vgz
        self.fsq = fsq
        self.CHR = CHR
        self.BP0 = BP0
        self.NBP = NBP

        ## actuall number of basepairs to fetch
        self.__nbp__ = 0xFFFFFFFF if NBP is None or NBP < 1 else nbp
        
        ## link to the *.fastq
        self.__ref__ = Fasta(fsq, key_fn = lambda x: x.split()[0])

        ## link to the *.vcf.gz
        self.__vcf__ = vcf.Reader(filename = vgz)
        if CHR is not None:
            self.__vcf__ = self.__vcf__.fetch(self.CHR, self.BP0, BP1)
            
        ## sample size
        self.__ssz__ = len(self.__vcf__.samples)

        ## previous site
        self.__pos__ = 0 if BP0 is None else BP0

        ## the current locus
        try:
            self.__gvr__  = next(self.__vcf__)
        except StopIteration as e:
            self.__gvr__  = e

    def __next__(self):
        ## reach the wanted number of nucleotide
        if self.__cnt__ == self.__nbp__:
            raise StopIteration()

        ret = None
        
        ## are we going through a variant?
        while(self.__gvr__ and self.__pos__ == self.__gvr__.POS):
            ## the next nucleotide is in the variant
            try:
                ret = next(self.__gvr__)
                break
            ## advance pointer for the reference sequence
            except StopIteration:
                self.__pos__ += 1

            ## find the next variant
            try:
                self.__gvr__  = next(self.__vcf__)
            ## announce there is no more variant
            except StopIteration as e:
                self.__gvr__  = None

        ## the next nucleotide is in the reference sequence
        if ret is None:
            ## chromosome
            c = self.__ref__[self.__vcf__.CHROM]
            g = c[self.__pos__]

            if(isinstance(self.__gvr__, StopIteration):
        elif isinstance(self.__gvr__, StopIteration):
            pass
        else:
            pass
        ## if self.__loc__ is None or self.__loc__.POS == self.__pos__ 
        ## self.__this__ = next(self.__gno__)
    
    
if __name__ == '__main__':
    ## add project root to python path
    import sys

    ## human reference genome build 37, patch 5
    ## fsq = load_ref('../raw/hs37d5.fa')

    ## 1000 genome data
    g1k = vcf.Reader(filename = '../raw/hs37d5/22.vcf.gz')


