# draw genome samples
d=raw/W08
hpcwp "python -c 'import dat; dat.rseq(256, \"{n:04X}\")'" -d $d -i20000 -q400 -n1 --wtm 4 -m2 --cp src/dat.py --ln src/gsq --ln raw/wgs

d=raw/H08
hpcwp "python -c 'import dat; dat.rseq(256, \"haf\", \"{n:04X}\")'" -d $d -i20000 -q400 -n1 --wtm 4 -m2 --cp src/dat.py --ln src/gsq --ln raw/haf

# pre-train, 20 epoch, overfit by 4 folds in 2 layers
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
cp="src/rdm/gdy.py,src/rdm/hlp.py,src/rdm/exb.py"
s=raw/W09; d=sim/W09/10_PTN/;
for f in $s/*.npz; do
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/10_{i:03d}' python -c 'import rdm.gdy; rdm.gdy.main(\"$f\", usb=.2, ugv=.5, cvk=3)'"
done | head -n 2000 | hpcwp - -d $d -q10 -p4 -t.2 -m1 --cp "$cp" --ln src/rdm,raw --pfx "$px" --md "$md"

# fine-tune the 1 st layer (-2 overfit)
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
cp="src/rdm/ftn.py,src/rdm/hlp.py,src/rdm/exb.py"
s=sim/W09/10_PTN; d=sim/W09/20_FT1
for f in $s/*.pgz
do
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/20_{i:03d}' python -c 'import rdm.ftn; rdm.ftn.main(\"$f\", nft=500, ae1=1, ovr=1)'"
done | hpcwp - -d $d -q10 -p4 -t.2 -m1 --cp "$cp" --ln src/rdm,sim --pfx "$px" --md "$md"

# fine-tune the deeper layer (-0 overfit)
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
cp="src/rdm/ftn.py,src/rdm/hlp.py,src/rdm/exb.py"
sdp=5; ddp=6
s=sim/W09/20_FT$sdp; d=sim/W09/20_FT$ddp
for f in $s/*.pgz
do
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/20_{i:03d}' python -c 'import rdm.ftn; rdm.ftn.main(\"$f\", nft=300, ae1=$ddp, ovr=1)'"
done | hpcwp - -d $d -q10 -p4 -t.3 -m2 --cp "$cp" --ln src/rdm,sim --pfx "$px" --md "$md"

# find out ANNs that did not not converge
cd $d;
grep -L "^NT[:] halted[.]$" log/* > ncv.txt
grep -f ncv.txt -h pbs/* | sed 's/.*\(python[^\&]*\).*$/\1/; s/nft=[0-9][0-9]*/nft=1000/' > ncv.cmd
rm std/* pbs/* log/*
cd $p
n=1; while read f; do
    i=$(printf "%03d" $n)
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/20_$i'" $f
    n=$[n+1]
done < $d/ncv.cmd | hpcwp - -d$d -q1 -p4 -t2 -m2 --cp "$cp" --ln src/rdm,sim --pfx "$px" --md "$md"

# export training data, and high order features
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
cp="src/rdm/ftn.py,src/rdm/hlp.py,src/rdm/exb.py"
es=sim/W08/20_FT5; ed=sim/W08/30_HW5
for f in $es/*.pgz
do
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/30_{i:03d}' python -c 'import rdm.ftn; rdm.ftn.ept(\"$f\")'"
done | hpcwp - -d $ed -q100 -p1 -t.015 -m1 --cp "$cp" --ln src/rdm,sim --pfx "$px" --md "$md"

# simple network test
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
s=raw/W08/21_FT3; d=raw/W08/21_FT4
seq 01000 01500 | xargs printf %x'\n' | while read i
do
    if [ -e "$d/$i.tgz" ]; then continue; fi
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/64_{i:03d}' python -c 'import sim; sim.main(\"$s\", \"$i\")'"
done | hpcwp - -d $d -q1 -p4 -t2 -m2 --cp src/sim.py --ln src/rdm,raw --pfx "$px" --md "$md"

md="GNU/4.9,R/3.3.0"
s0=sim/W08/30_HW5
s1=sim/W08/31_HW5
d1=sim/W08/41_CV5
rc="time Rscript -e 'source(\"sim.R\"); main(\"$s0\", \"$s1\", \"{n:03d}\", r2=.05, efr=.05, ssz=(1:5)*100, n.i=20, wdp=0:5)'"
hpcwp "${rc}" -i 200 -d${d1} -q1 -n1 -t4 -m4 --cp src/sim.R --ln src,sim --md $md
