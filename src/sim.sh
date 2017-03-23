# draw genome samples
d=raw/W09
hpcwp "python -c 'import dat; dat.rseq(512, \"wgs\", \"{n:04X}\")'" -d $d -i10000 -q200 -n1 --wtm 4 -m2 --cp src/dat.py --ln src/gsq --ln raw/wgs

# fine-tune the deeper layer (-0 overfit)
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
cp="src/rdm/hlp.py,src/rdm/exb.py"
tf="THEANO_FLAGS='base_compiledir=$HOME/TC/RRS_DP4_{i:03d},device=cpu'"
p=$(pwd); s=raw/W09; d=sim/W09/RSS_DP4
for f in $s/*z
do
    # echo "time $tf python -c 'from rdm.ftn_hg import main; main(\"$f\", nft=300, hte=5e-5, reg=\"L2\", err=\"BH\", wdp=6, ovr=1)'"
    echo "time $tf python -c 'from rdm.cv_ftn_rss import main; main(\"$f\", nft=50, hvp=50, acc=1.02, dec=.95, wdp=6, ovr=1)'"
done | head -n 2000 | hpcwp - -d $d -q10 -p4 -t.3 -m2 --cp "$cp" --ln src/rdm,raw --pfx "$px" --md "$md"
# echo "time $tf python -c 'from rdm.cv_ftn import main; main(\"$f\", nft=100, hvp=50, wdp=5, ovr=1)'"
# echo "time $tf python -c 'from rdm.cv_ftn_relu import main; main(\"$f\", nft=100, lrt=5e-4, hvp=50, acc=1.02, dec=.95, wdp=5, ovr=1)'"
# echo "time $tf python -c 'from rdm.ftn_hg import main; main(\"$f\", nft=300, hte=5e-5, reg=\"L2\", err=\"BH\", wdp=6, ovr=1)'"

# find out ANNs that did not not converge
cd $d;
grep -L "^NT[:] halted[.]$" log/* > ncv.txt; grep -f ncv.txt -h pbs/* | sed 's/.*\(python[^\&]*\).*$/\1/; s/nft=[0-9][0-9]*/nft=1000/' > ncv.cmd
# rm std/* pbs/* log/*
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
tf="THEANO_FLAGS='base_compiledir=$HOME/TC/RRS_DP5_{i:03d},device=cpu'"
while read f; do
    echo "time $tf $f"
done < ncv.cmd | hpcwp - -d./ -q1 -p4 -t2 -m2 --pfx "$px" --md "$md"

# export training data, and high order features
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"
md="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"
cp="src/rdm/ftn.py,src/rdm/hlp.py,src/rdm/exb.py"
tf="THEANO_FLAGS='base_compiledir=$HOME/TC/30_{i:03d},device=cpu'"
s=sim/W09/RSS_DP5 ; d=sim/W09/RSS_EX5
for f in $s/*.pgz
do
    echo "time $tf python -c 'from rdm.cv_ftn_relu import ept; ept(\"$f\")'"
done | hpcwp - -d $d -q100 -p1 -t.015 -m1 --cp "$cp" --ln src/rdm,sim --pfx "$px" --md "$md"

md="GNU/4.9,R/3.3.0"
s0=sim/W09/SSS_EX5
d1=sim/W09/SSS_S01
rc="time Rscript -e 'source(\"sim.R\"); main(\"$s0\", \"{n:03d}\", r2=.05, efr=.05, eft=c(\"gno\"), ssz=(2:9)*100, n.i=20)'"
hpcwp "${rc}" -i 100 -d${d1} -q1 -n1 -t1 -m2 --cp src/sim.R --ln src,sim --md $md

md="GNU/4.9,R/3.3.0"
s0=sim/W09/RRS_EX5
d1=sim/W09/RRS_S01
rc="time Rscript -e 'source(\"sim.R\"); main(\"$s0\", \"{n:03d}\", r2=.05, efr=.05, eft=c(\"gno\"), ssz=(2:9)*100, n.i=20)'"
hpcwp "${rc}" -i 100 -d${d1} -q1 -n1 -t1 -m2 --cp src/sim.R --ln src,sim --md $md

md="GNU/4.9,R/3.3.0"
s0=sim/W09/RSS_EX5
d1=sim/W09/RSS_S01
rc="time Rscript -e 'source(\"sim.R\"); main(\"$s0\", \"{n:03d}\", r2=.05, efr=.05, eft=c(\"gno\"), ssz=(2:9)*100, n.i=20)'"
hpcwp "${rc}" -i 100 -d${d1} -q1 -n1 -t1 -m2 --cp src/sim.R --ln src,sim --md $md

md="GNU/4.9,R/3.3.0"
s0=sim/W09/RRH_EX5
d1=sim/W09/RRH_S01
rc="time Rscript -e 'source(\"sim.R\"); main(\"$s0\", \"{n:03d}\", r2=.05, efr=.05, eft=c(\"gno\"), ssz=(2:9)*100, n.i=20)'"
hpcwp "${rc}" -i 100 -d${d1} -q1 -n1 -t1 -m2 --cp src/sim.R --ln src,sim --md $md
