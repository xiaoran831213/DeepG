# use 1000 samples
dp=1; s=raw/W09; d=sim/gmx/T0${dp}; rm $d/{pbs,std,log}/*; tf="THEANO_FLAGS='base_compiledir=$HOME/TC/TE${dp}_{a:03d},device=cpu'"
# while read f; do
#     echo "time $tf python -c 'from ftn import main; main(\"$s/$f.npz\", nep=500, wdp=${dp}, nsb=1000, hte=1e-6, ovr=1, htg=1e-9, eot=1)' > log/$(basename ${f} .npz)"
# done < $d/ncv.txt | hpcwp - -d $d -q1 -p4 -t.50 -m2 --cp "src/ftn.py" --tag ${d##*/} --ln sim,raw --log None
for f in $s/0*.npz
do
    echo "time $tf python -c 'from ftn import main; main(\"$f\", nep=500, wdp=${dp}, nsb=1000, hte=1e-6, ovr=1, htg=1e-9, eot=1)' > log/$(basename ${f} .npz)"
done | hpcwp - -d $d -a4 -q8 -p4 -t.25 -m2 --cp "src/ftn.py" --tag ${d##*/} --ln sim,raw --log None

dp=5; s=sim/gmx/T0$[dp-1]; d=sim/gmx/T0${dp}; rm $d/{pbs,std,log}/*; tf="THEANO_FLAGS='base_compiledir=TMPDIR/TE${dp}_{a:03d},device=cpu'"
while read f; do
    echo "time $tf python -c 'from ftn import main; main(\"$f\", nep=2400, pep=300, wd0=$[dp-1], wdp=${dp}, eot=1, hte=1e-6, ovr=1, htg=1e-9)' > log/$(basename ${f} .pgz)"
done < $d/ncv.txt | hpcwp - -d $d -a5 -q1 -p4 -t2 -m2 --cp "src/ftn.py" --tag ${d##*/} --ln sim,raw --log None
# for f in $s/0*.pgz
# do
#     echo "time $tf python -c 'from ftn import main; main(\"$f\", nep=400, pep=300, wd0=$[dp-1], wdp=${dp}, eot=1, hte=1e-6, ovr=1, htg=1e-9)' > log/$(basename ${f} .pgz)"
# done | hpcwp - -d $d -a4 -q8 -p4 -t.3 -m2 --cp "src/ftn.py" --tag ${d##*/} --ln sim --log None

dp=5; s=sim/gmx/T0${dp}; d=sim/gmx/TE${dp}; rm $d/{pbs,std,log}/*; eot=1000
for f in $s/*.pgz; do
    o=${f##*/}; o=${o%%.*}
    echo "python -c 'from xutl import *; x=lpz(\"$f\"); xpt(\"$o\", **x) if x[\"eot\"] < $eot else None'"
    echo "if [ -e $o ]; then tar -zcf $o.tgz $o --remove-files; fi"
done | hpcwp - -d${d} -q512 -p1 --wtm 2 -m2 --ln sim --tag ${d##*/} --log None

# merged
ml="GNU/4.9"; s=sim/gmx; d=sim/gmx/TEM; rm $d/{log,pbs,std}/*;
for f in $(printf "%04X\n" {1..4096}); do
    echo "Rscript -e 'source(\"gmy.R\"); fs=c(\"${s}/TE1/${f}.tgz\", \"${s}/TE2/${f}.tgz\", \"${s}/TE3/${f}.tgz\", \"${s}/TE4/${f}.tgz\", \"${s}/TE5/${f}.tgz\"); m=merge(fs); saveRDS(m, \"${f}.rds\")'"
done | hpcwp - -d${d} -q512 -b1 --wtm 4 -m2 --cp src/gmy.R --tag ${d##*/} --ln src,sim --ml $ml --log none

# gaussian link, 4 types of bases
ml="GNU/4.9"; d=sim/rsq_g4b; rm $d/{log,pbs,std}/*;
sfx1="Rscript -e 'f=dir(\".\", \"^{i:03d}\"); d=do.call(rbind, lapply(f, readRDS)); saveRDS(d, \"{i:03d}.rds\")'"
sfx2="rm {i:03d}_???.rds"
for sq in $(printf "%04X\n" {1..4096} | sort -R | head -n 1024); do
    printf "%s\n" sim/gmx/TEM/${sq}.rds" "1000" "{.01,.02,.05,.10,.15}" "gau" "{g,g:g[],'I(g^2)','g*g[]'} | while read sc sz r2 fm md; do
	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{i:03d}_{j:d}{k:2d}.rds\", ssz=${sz}, max.gvr=64, r2=${r2}, rep=1, fam=${fm}, mdl=~${md})'"
    done
done | hpcwp - -d${d} -q20 -b4 --wtm 3 -m4 --cp src/gmy.R --tag ${d##*/} --ln src,sim --ml $ml --log none --sfx "$sfx1" --sfx "$sfx2"

# parity
ml="GNU/4.9"; d=sim/gmx/s4e; rm $d/{log,pbs,std}/*;
for sq in $(printf "%04X\n" {1..4096} | sort -R | head -n 1000); do
    printf "%s\n" sim/gmx/TEM/${sq}.rds" "1000" "{.05,.10,.20,.50,1.0}" "gau" "{g,g:g[],g^2,g[]*g[]} | while read sc sz r2 fm md; do
      	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{n:04X}.rds\", ssz=${sz}, max.gvr=128, r2=${r2}, rep=1, fam=${fm}, mdl=~(${md}):p)'"
    done
done | hpcwp - -d${d} -q20 -b4 --wtm 2 -m2 --cp src/gmy.R --tag ${d##*/} --ln src,sim --ml $ml --log none

ml="GNU/4.9"; d=sim/rsq_fpr; rm $d/{log,pbs,std}/*;
for sq in $(printf "%04X\n" {1..4096} | sort -R | head -n 1024); do
    printf "%s\n" sim/gmx/TEM/${sq}.rds" "1000" "{.01,.02,.05,.10,.20,.50}" "gau" "'g[]*g[]' | while read sc sz r2 fm md; do
	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{n:04X}.rds\", ssz=${sz}, max.gvr=64, r2=${r2}, rep=1, fpr=0x04, fam=${fm}, mdl=~(${md}):p)'"
	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{n:04X}.rds\", ssz=${sz}, max.gvr=64, r2=${r2}, rep=1, fpr=0x08, fam=${fm}, mdl=~(${md}):p)'"
	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{n:04X}.rds\", ssz=${sz}, max.gvr=64, r2=${r2}, rep=1, fpr=0x10, fam=${fm}, mdl=~(${md}):p)'"
	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{n:04X}.rds\", ssz=${sz}, max.gvr=64, r2=${r2}, rep=1, fpr=0x20, fam=${fm}, mdl=~(${md}):p)'"
    done
done | hpcwp - -d${d} -q20 -b4 --wtm 3 -m2 --cp src/gmy.R --tag ${d##*/} --ln src,sim --ml $ml --log none

# mix
ml="GNU/4.9"; d=sim/rsq_mix; rm $d/{log,pbs,std}/*;
sfx1="Rscript -e 'f=dir(\".\", \"^{i:03d}\"); d=do.call(rbind, lapply(f, readRDS)); saveRDS(d, \"{i:03d}.rds\")'"
sfx2="rm {i:03d}_???.rds"
for sq in $(printf "%04X\n" {1..4096} | sort -R | head -n 1024); do
    printf "%s\n" sim/gmx/TEM/${sq}.rds" "1000" "{.01,.02,.05,.10,.20,.50} | while read sc sz r2; do
	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{i:03d}_{j:d}{k:02d}.rds\", ssz=${sz}, max.gvr=64, r2=${r2}, rep=1, fam=gau, mdl=~g,   fpr=0)'"
	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{i:03d}_{j:d}{k:02d}.rds\", ssz=${sz}, max.gvr=64, r2=${r2}, rep=1, fam=gau, mdl=~g*g, fpr=0)'"
	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{i:03d}_{j:d}{k:02d}.rds\", ssz=${sz}, max.gvr=64, r2=${r2}, rep=1, fam=sin, mdl=~g,   fpr=0)'"
	echo "Rscript -e 'source(\"gmy.R\"); main(\"${sc}\", \"{i:03d}_{j:d}{k:02d}.rds\", ssz=${sz}, max.gvr=64, r2=${r2}, rep=1, fam=gau, mdl=~p,   fpr=0x10)'"
    done
done | hpcwp - -d${d} -q20 -b4 --wtm 3 -m8 --cp src/gmy.R --tag ${d##*/} --ln src,sim --ml $ml --log none --sfx "$sfx1" --sfx "$sfx2"
