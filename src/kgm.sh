d=$SCR/deep/1kg; mkdir -p $d; ln -s $d

# extract MAF>=0.005, remove INFO, convert to plink binary, keep only SNPs.
s=$HOME/study/1kg; d=1kg/chr; mkdir -p $d;
for i in $(seq -w 1 22)
do
    echo "bcftools view $s/$i.vcf.gz -i 'MAF>0.00499' -v snps | bcftools annotate -x INFO -Oz -o $i.vcf.gz"
    echo "bcftools index -t $i.vcf.gz"
    echo "plink --vcf $i.vcf.gz --snps-only --make-bed --out $i --memory 4032"
done | hpcwp - -d$d -t1 -m4 -q2

# merge into one, convert to plink binary, extract genomic map
hpcwp 'time bcftools concat -o ../one.vcf.gz -Oz $(ls *.vcf.gz)' -d 1kg/chr -t8 -m1 -q1
bcftools index -t 1kg/one.vcf.gz
bcftools view -GH 1kg/one.vcf.gz | cut -f1-5 | gzip > 1kg/one.map.gz
# plink --vcf $d/cc.vcf.gz --make-bed --out $d/cc --memory 4096

# randomly sample from the gnomes without replacement, forming slices of 2^14 variants
# 1) create randome map
d=1kg/rnd; mkdir -p $d; sz=$((2**14))
zcat 1kg/one.map.gz | cut -f1,2 | sort -R | split -a4 -l $sz -d - "$d/"
for s in $d/[0-9][0-9][0-9][0-9]; do
    sort $s -k1,1n -k2,2n > $s.map; rm $s
done

# 2) slice the genome, convert the parts to both plink and numpy format.
d=1kg/rnd
for m in $d/*.map; do
    m=${m##*/}
    m=${m%%.*}
    if [ -e $d/$m.vcf.gz ]; then # vcf.gz
	echo "echo"
    else
	echo "bcftools view -R ${m}.map -Oz -o ${m}.vcf.gz one.vcf.gz"
    fi

    if [ -e $d/$m.bed ]; then
	echo "echo"
    else
	echo "plink --vcf ${m}.vcf.gz --make-bed --out ${m}"
    fi
    if [ -e $dst/$m.npz ]; then
	echo "echo"
    else
	echo "time python -c 'from vsq import vcf2npz as main; main(\"${m}.vcf.gz\")'"
    fi
done | hpcwp - -d$d -q3 -a20 --wtm 1 -m4 --tag npz --ln 1kg/one.vcf.gz,1kg/one.vcf.gz.tbi --cp src/gsq/vsq.py --log none


# gmx versus pcs
s=1kg/rnd; d=1kg/1x1_ax1; mkdir -p $d;rm $d/{pbs,std,log}/*; tf="THEANO_FLAGS='base_compiledir=TMP/{a:03d},device=cpu'"
cfg='prg=0, seed=1, N=1000, P=500, rsq=.5, frq=.5, fam="gau", mdl="a*a", nep=3000, lr=1e-3, hvp=100'
# for f in $s/*.npz; do
ls -1 $s/*.npz | head -n 100 | while read f; do
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1000]*1, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1000]*2, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1000]*3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1000]*4, gtp=\"flt\", xtp=\"gmx\")'"

    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1500]*1, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1500]*2, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1500]*3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1500]*4, gtp=\"flt\", xtp=\"gmx\")'"
    
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*1, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*2, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*4, gtp=\"flt\", xtp=\"gmx\")'"

    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[2000]*1, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[2000]*2, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[2000]*3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[2000]*4, gtp=\"flt\", xtp=\"gmx\")'"
done | hpcwp - -d$d -p4 --wtm 4 -q1 -m24 --cp src/kgm.py,src/gsm.py,src/bmk.py --ln 1kg --tag ${d##*/}

# binomial family
s=1kg/rnd; d=1kg/1x1_bx1; mkdir -p $d;rm $d/{pbs,std,log}/*; tf="THEANO_FLAGS='base_compiledir=TMP/{a:03d},device=cpu'"
cfg='seed=1, N=625, P=3000, rsq=1.0, frq=.50, fam="bin", mdl="g*g", nptr=[], svn=False, prg=0, hvp=200'
# for f in $s/*.npz; do
ls -1 $s/*.npz | head -n 100 | while read f; do
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1000]*1, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1000]*2, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1000]*3, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1000]*4, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"

    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1500]*1, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1500]*2, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1500]*3, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[1500]*4, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"

    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*1, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*2, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*3, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*4, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"

    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[2000]*1, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[2000]*2, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[2000]*3, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[2000]*4, nep=5000, lr=1e-3, gtp=\"flt\", xtp=\"gmx\")'"
done | hpcwp - -d$d -p4 --wtm 4 -q3 -m24 --cp src/kgm.py,src/gsm.py,src/bmk.py --ln 1kg --tag ${d##*/}

# sin family
s=1kg/rnd; d=1kg/sin_000; mkdir -p $d;rm $d/{pbs,std,log}/*; tf="THEANO_FLAGS='base_compiledir=~/TC/SIN_{a:03d},device=cpu'"
cfg='seed=1, N=625, P=2000, rsq=1.0, frq=.20, fam="sin1.5", mdl="g", prg=0, hvp=50'
# for f in $s/*.npz; do
ls -1 $s/*.npz | head -n 50 | while read f; do
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[2000] + [1000]*3, nep=2000, lr=5e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[3000] + [1000]*3, nep=2000, lr=5e-3, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[4000] + [1000]*3, nep=2000, lr=5e-3, gtp=\"flt\", xtp=\"gmx\")'"
done | hpcwp - -d$d -p4 --wtm 4 -q3 -m4 --cp src/kgm.py,src/gsm.py,src/bmk.py --ln 1kg --tag ${d##*/}

# parity * 8
s=1kg/rnd; d=1kg/1x1_pa2; mkdir -p $d;rm $d/{pbs,std,log}/*; tf="THEANO_FLAGS='base_compiledir=~/TC/PAR_{a:03d},device=cpu'"
# for f in $s/*.npz; do
ls -1 $s/*.npz | head -n 100 | while read f; do
    cfg='seed=1, N=625, P=3000, rsq=1.0, frq=.50, fam="gau", mdl="p2", hvp=200, prg=0'
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*1, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*2, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*3, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*4, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*5, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*6, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*7, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"

    cfg='seed=1, N=625, P=3000, rsq=1.0, frq=.50, fam="gau", mdl="p3", hvp=200, prg=0'
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*1, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*2, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*3, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*4, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*5, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*6, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*7, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"

    cfg='seed=1, N=625, P=3000, rsq=1.0, frq=.50, fam="gau", mdl="p4", hvp=200, prg=0'
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*1, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*2, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*3, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*4, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*5, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*6, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*7, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"

    cfg='seed=1, N=625, P=3000, rsq=1.0, frq=.50, fam="gau", mdl="p5", hvp=200, prg=0'
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*1, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*2, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*3, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*4, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*5, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*6, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
    echo "time $tf python -c 'from kgm import main; main(\"$f\", sav=\"{n:04d}\", $cfg, dim=[ 500]*7, nep=3000, lr=3e-5, gtp=\"flt\", xtp=\"gmx\")'"
done | hpcwp - -d$d -p4 --wtm 4 -q7 -m8 --cp src/kgm.py --ln 1kg,src/gsm.py,src/bmk.py --tag ${d##*/}
