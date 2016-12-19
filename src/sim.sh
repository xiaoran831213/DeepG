# draw genome samples
d=raw/W08
hpcwp "python -c 'import dat; dat.rseq(256, \"{n:04X}\")'" -d $d -i20000 -q400 -n1 --wtm 4 -m2 --cp src/dat.py --ln src/gsq --ln raw/wgs

d=raw/H08
hpcwp "python -c 'import dat; dat.rseq(256, \"haf\", \"{n:04X}\")'" -d $d -i20000 -q400 -n1 --wtm 4 -m2 --cp src/dat.py --ln src/gsq --ln raw/haf

# pre-train, 20 epoch, overfit by 4 folds in 2 layers
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
s=raw/W08/00_GNO; d=sim/W08/10_PTN/;
for f in $s/*.npz
do
    f=${f##*/}
    i=${f%%.*}
    if [ -e "$d/$i.pgz" ]; then continue; fi
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/10_{i:03d}' python -c 'import rdm.gdy; rdm.gdy.main(\"$s/$f\")'"
done | head -n 1000 | hpcwp - -d $d -q10 -p4 -t.2 -m3 --cp src/rdm/gdy.py --ln src/rdm,raw --pfx "$px" --md "$md"

# fine-tune the 1 st layer (-2 overfit)
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
s=sim/W08/10_PTN; d=sim/W08/20_F-2
for f in $s/*.pgz
do
    # if [ -e "$d/${f##*/}" ]; then continue; fi
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/20_{i:03d}' python -c 'import rdm.ftn; rdm.ftn.main(\"$f\", nft=100, ae1=1, ovr=1)'"
done | hpcwp - -d $d -q5 -p4 -t.3 -m1.5 --cp src/rdm/ftn.py --ln src/rdm,sim --pfx "$px" --md "$md"

# fine-tune the 3rd layer (-0 overfit)
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
s=sim/W08/20_F+1; d=sim/W08/20_F+2
for f in $s/*.pgz
do
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/20_{i:03d}' python -c 'import rdm.ftn; rdm.ftn.main(\"$f\", nft=100, ae1=5, ovr=1)'"
done | hpcwp - -d $d -q5 -p4 -t.7 -m1.5 --cp src/rdm/ftn.py --ln src/rdm,sim --pfx "$px" --md "$md"

# simple network test
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
s=raw/W08/00_GNO; d=raw/W08/1K_FTN
seq 01000 01500 | xargs printf %x'\n' | while read i
do
    if [ -e "$d/$i.tgz" ]; then continue; fi
    echo "time THEANO_FLAGS='base_compiledir=$HOME/TC/64_{i:03d}' python -c 'import sim; sim.main(\"$s\", \"$i\")'"
done | hpcwp - -d $d -q1 -p4 -t2 -m2 --cp src/sim.py --ln src/rdm,raw --pfx "$px" --md "$md"

md="GNU/4.9,R/3.3.0"
for q in {_32,_64,128,256,512,1k_}; do
    s=raw/W08/${q}FTN
    d=raw/W08/${q}HWU
    c="time Rscript -e 'source(\"sim.R\"); main(\"$s\", \"{n:03d}\", ssz=(1:5)*200, n.i=20, use=c(\"m\", \"v\"))'"
    hpcwp "${c}" -i 100 -d${d} -q1 -n1 -t2 -m4 --cp src/sim.R --ln src,raw --md $md
done
