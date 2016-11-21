# draw genome samples
d=raw/W08
hpcwp "python -c 'import dat; dat.rseq(256, \"{n:04X}\")'" -d $d -i20000 -q400 -n1 --wtm 4 -m2 --cp src/dat.py --ln src/gsq --ln raw/wgs

d=raw/H08
hpcwp "python -c 'import dat; dat.rseq(256, \"haf\", \"{n:04X}\")'" -d $d -i20000 -q400 -n1 --wtm 4 -m2 --cp src/dat.py --ln src/gsq --ln raw/haf

# rm -rf /mnt/home/tongxia1/.theano/*
s=raw/W08/00_GNO; d=sim/001
seq -w 1 500 | while read i
do
    if [ -e "$d/$i.tgz" ]
    then
	continue
    fi
    echo "time python -c 'import ts6; ts6.main(\"$s\", \"$i\", npt=20, nft=300)'"
done | hpcwp - -d $d -q1 -p4 -n1 -t4 -m3 --cp src/ts6.py --ln src/rdm --ln raw --pfx 'export MKL_NUM_THREADS=4' --md 'GNU/4.9' --md 'Python/2.7.2' --md 'openblas/0.2.15' --md 'SciPy/0.13.0'

c="Rscript -e 'source(\"sim.R\"); main(\"sim/$s\", \"{n:03d}\", ssz=(1:9)*100, eff=c(\"d\", \"n\"))'"
s=H08_U10_D05 d=H08_U10_D05_POW
hpcwp "time Rscript -e 'source(\"sim.R\"); main(\"sim/$s\", \"{n:03d}\", ssz=(1:10)*200, n.i=20)'" -i 200 -d sim/$d -q1 -n1 -t4 -m4 --cp src/sim.R --ln src --ln sim --md GNU/4.9 --md R/3.3.0
