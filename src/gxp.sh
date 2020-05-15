px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"; ml="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"; tf="THEANO_FLAGS='base_compiledir=$SCR/TC/GXP_D04_{a:03d},device=cpu'"
p=$(pwd); s=raw/GXP/01; d=sim/GXP/D04/01; rm $d/{pbs,std,log}/*
for f in $s/*.vcf.gz
do
    # if [ -e $d/$(basename $f .vcf.gz).pgz ]; then continue; fi
    echo "time $tf python -c 'from gxp import main; main(\"$f\", gdy=0, nep=50, wdp=4, ovr=1, hlt=0, pca=1, xpt=1)'"
done | hpcwp - -d $d -a10 -q1 -p4 -t2 -m12 --cp "src/gxp.py" --tag D04 --ln raw --pfx "$px" --mld "$ml" --log "{a:03}_{i:03}"

px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"; ml="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"; tf="THEANO_FLAGS='base_compiledir=$SCR/TC/GXP_D06_{a:03d},device=cpu'"
p=$(pwd); s=raw/GXP/01; d=sim/GXP/D06/01; rm $d/{pbs,std,log}/*
for f in $s/*.vcf.gz; do
    echo "time $tf python -c 'from gxp import main; main(\"$f\", gdy=0, nep=0, wdp=6, ovr=1, hlt=0, pca=1, xpt=1)'"
done | hpcwp - -d $d -a10 -q1 -p4 -t2 -m12 --cp "src/gxp.py" --tag D06 --ln raw --pfx "$px" --mld "$ml" --log "{a:03}_{i:03}"

# training
px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"; ml="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"; tf="THEANO_FLAGS='base_compiledir=$SCR/TC/GXP_D08_{a:03d},device=cpu'"
p=$(pwd); s=raw/GXP/01; d=sim/GXP/D08/01; rm $d/{pbs,std,log}/*
for f in $s/*.vcf.gz; do
    echo "time $tf python -c 'from gxp import main; main(\"$f\", gdy=0, nep=300, wdp=8, ovr=1, pca=1, xpt=1)'"
done | hpcwp - -d $d -a10 -q1 -p4 -t4 -m12 --cp "src/gxp.py" --tag D08 --ln raw --pfx "$px" --mld "$ml" --log "{a:03}_{i:03}"
# for f in $s/*.txt; cp $f $d/${s%%.txt}.emx; done

px="export MKL_NUM_THREADS=4;OMP_NUM_THREADS=4"; ml="GNU/4.9,Python/2.7.2,openblas/0.2.15,SciPy/0.13.0"; tf="THEANO_FLAGS='base_compiledir=$SCR/TC/GXP_D10_{a:03d},device=cpu'"
p=$(pwd); s=raw/GXP/01; d=sim/GXP/D10/01; rm $d/{pbs,std,log}/*
for f in $s/*.vcf.gz; do
    echo "time $tf python -c 'from gxp import main; main(\"$f\", gdy=0, nep=200, wdp=10, ovr=1, pca=1, xpt=1)'"
done | hpcwp - -d $d -a10 -q1 -p4 -t2 -m12 --cp "src/gxp.py" --tag D10 --ln raw --pfx "$px" --mld "$ml" --log "{a:03}_{i:03}"
# for f in $s/*.txt; cp $f $d/${s%%.txt}.emx; done

ml="GNU/4.9,R/3.3.0"; s=sim/GXP/D06/01; d=sim/00I; cm="Rscript -e 'source(\"gxp.R\"); main(\"$s\", \"{n:03d}.rds\", r2=1.0, rep=50, dst=\"gau\", ecp=\"gen\")'"
hpcwp "${cm}" -i 100 -d${d} -q1 -t.2 -m8 --cp src/gxp.R --ln src,dat,sim  --ml $ml
ml="GNU/4.9,R/3.3.0"; s=sim/GXP/D06/01; d=sim/10I; cm="Rscript -e 'source(\"gxp.R\"); main(\"$s\", \"{n:03d}.rds\", r2=1.0, rep=50, dst=\"bin\", ecp=\"gen\")'"
hpcwp "${cm}" -i 100 -d${d} -q1 -t.2 -m8 --cp src/gxp.R --ln src,dat,sim  --ml $ml
ml="GNU/4.9,R/3.3.0"; s=sim/GXP/D06/01; d=sim/20I; cm="Rscript -e 'source(\"gxp.R\"); main(\"$s\", \"{n:03d}.rds\", r2=1.0, rep=50, dst=\"poi\", ecp=\"gen\")'"
hpcwp "${cm}" -i 100 -d${d} -q1 -t.2 -m8 --cp src/gxp.R --ln src,dat,sim  --ml $ml

ml="GNU/4.9,R/3.3.0"; s=sim/GXP/D06/01; d=sim/010; cm="Rscript -e 'source(\"gxp.R\"); main(\"$s\", \"{n:03d}.rds\", r2=1.0, rep=50, dst=\"gau\", ecp=\"med\")'"
hpcwp "${cm}" -i 100 -d${d} -q1 -t.2 -m8 --cp src/gxp.R --ln src,dat,sim  --ml $ml
ml="GNU/4.9,R/3.3.0"; s=sim/GXP/D06/01; d=sim/110; cm="Rscript -e 'source(\"gxp.R\"); main(\"$s\", \"{n:03d}.rds\", r2=1.0, rep=50, dst=\"bin\", ecp=\"med\")'"
hpcwp "${cm}" -i 100 -d${d} -q1 -t.2 -m8 --cp src/gxp.R --ln src,dat,sim  --ml $ml
ml="GNU/4.9,R/3.3.0"; s=sim/GXP/D06/01; d=sim/210; cm="Rscript -e 'source(\"gxp.R\"); main(\"$s\", \"{n:03d}.rds\", r2=1.0, rep=50, dst=\"poi\", ecp=\"med\")'"
hpcwp "${cm}" -i 100 -d${d} -q1 -t.2 -m8 --cp src/gxp.R --ln src,dat,sim  --ml $ml

ml="GNU/4.9,R/3.3.0"; s=sim/GXP/D04/01; d=sim/500
printf "%s\n" {1,2,3}" "{gen,med}" "{gau,bin} | while read w ecp dst
do
    echo "Rscript -e 'source(\"gxp.R\"); main(\"${s}\", \"{n:03d}.rds\", r2=1.0, wdp=${w}, rep=100, ecp=\"${ecp}\", dst=\"${dst}\")'"
done | hpcwp - -i 50 -d${d} -q20 -t.1 -m8 --cp src/gxp.R --ln src,dat,sim  --ml $ml
