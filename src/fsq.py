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

## FASTQ chromosome keys
CKY = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', 'X', 'Y']
    
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

    def rewind(self):
        self.__pos__ = 0

class VcfSeq:
    """
    A iterator class to recover FASTQ sequences of all subjects
    from a VCF file  and its genotyping reference genome.
    """
    def __init__(self, vgz, fsq, CHR = None, BP0 = None, BP1 = None, NBP = None):
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
        self.BP1 = BP1
        self.NBP = NBP

        
        ## link to the *.vcf.gz
        vgz = vcf.Reader(filename = vgz)

        ## link to the *.fastq
        fsq = Fasta(fsq, key_fn = lambda x: x.split()[0])

        ## the chromosome, starting and closing positions
        ch = 0 if CHR is None else CHR
        b0 = 0 if BP0 is None else BP0
        b1 = len(fsq[CKY[ch]])
        if BP1 is not None:
            b1 = min(b1, BP1)
        
        ## retrict VCF range    
        vgz = vgz.fetch(ch, b0, b1)
            
        ## sample size
        self.__ssz__ = len(self.__vcf__.samples)


        ## chromosome, 0 based
        self.__chr__ = if CHR is None fsq[CKY[0]] else fsq[CKY[CHR]]
        
        ## starting position, 0 based
        self.__bp0__ = 0 if BP0 is None else BP0
        ## ending position, 0 based, exclusive
        self.__bp1__ = 

        ## current position, 0 based
        self.__pos__ = 0 if BP0 is None else BP0

        ## the current locus
        try:
            self.__gvr__  = next(self.__vcf__)
        except StopIteration as e:
            self.__gvr__  = e

    def __advc__(self):
        ## presumptively move the pointer one nucleotide forward
        p = self.__pos__ + 1
        c = self.__chr__

        ## reaching the ending position
        if c == self.CHR and p == self.BP1
            return None
        
        ## advance to the next chromosome
        if p == len(self.__ref__[CKY[c]]):
            c = c + 1
            p = 0

        ## reaching the end of the reference sequence
        if self.__chr__ == len(CKY):
            return None

        return True

    def __next__(self):
        ## reach the wanted number of nucleotide
        if self.__cnt__ == self.__nbp__:
            raise StopIteration()

        ret = None
        
        ## are we going through a variant?
        while(self.__gvr__
              and FCK[self.__chr__] == self.__gvr__.CHROM
              and self.__pos__ == self.__gvr__.POS):
            ## the next nucleotide is in the variant
            try:
                ret = next(self.__gvr__)
                break
            ## try advance pointer for the reference sequence
            except StopIteration:
                if not self.__advc__():
                    break
                
            ## find the next variant
            try:
                self.__gvr__  = next(self.__vcf__)
            ## announce there is no more variant
            except StopIteration as e:
                self.__gvr__  = None

        if ret is not None:
            return

        ## declare the end of reference sequence
            if self.__chr__ == len(FCK) or self.__chr__ == self.CHR
            
        ## the next nucleotide is in the reference sequence
        if ret is None:
            ## jump to the next chromosome if necessary
            if self.__pos__ == len(self.__ref__[FCK[self.__chr__]]):
                self.__chr__ += 1
                self.__pos__ == 0

            
            ret = ref_chr[self.__pos__] * self.__ssz__
                
            g = c[self.__pos__]

        ## if self.__loc__ is None or self.__loc__.POS == self.__pos__ 
        ## self.__this__ = next(self.__gno__)
    
    
if __name__ == '__main__':
    ## add project root to python path
    import sys

    ## 1000 genome data
    g1k = vcf.Reader(filename = '../raw/hs37d5/22.vcf.gz')







