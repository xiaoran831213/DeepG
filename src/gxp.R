source('src/dsg.R')
source('src/utl.R')
source('src/hlp.R')
source('src/hwu.R')
## library(MASS)

readSet <- function(fn)
{
    ## get file name stem
    fn <- sub('[.].*$', '', fn)
    fd <- dirname(fn)

    ## genomic matrix
    sbj <- scan(paste(fn, 'sbj', sep='.'), '', quiet=T)
    N <- length(sbj)
    gx0 <- scan(paste(fn, 'gx0', sep='.'), 0L, quiet=T)
    gx1 <- scan(paste(fn, 'gx1', sep='.'), 0L, quiet=T)
    P <- length(gx0) / N
    gmx <- array(
        c(gx0, gx1), c(P, N, 2L),
        list(gvr=sprintf('G%04X', 1:P), sbj=sbj, cpy=c('CP0', 'CP1')))

    ## high order features
    hof <- list()
    for(i in 1:9)
    {
        key <- paste('hf', i, sep='')
        fnk <- paste(fn, key, sep='.')
        if(!file.exists(fnk))
            break
        val <- scan(paste(fn, key, sep='.'), .0, quiet=T)
        Q <- length(val) / N
        val <- matrix(val, Q, N, F, list(ftr=sprintf('H%03X', 1:Q), sbj=sbj))
        hof[[key]] <- val
    }
    
    ## PCA
    pcs <- scan(paste(fn, 'pcs', sep='.'), .0, quiet=T)
    Q <- length(pcs) / N
    pcs <- matrix(pcs, Q, N, F, list(ftr=sprintf('P%03X', 1:Q), sbj=sbj))

    ## gene expressions
    emx.sbj <- scan(file.path(fd, 'emx.sbj'), '', quiet=T)
    N <- length(emx.sbj)
    emx <- scan(paste(fn, 'emx', sep='.'), '', quiet=T)
    P <- length(emx) / (N + 1)
    emx <- matrix(emx, P, N + 1, T)
    gen <- emx[, +1]
    emx <- as.numeric(emx[, -1])
    emx <- matrix(emx, P, N, T, list(gen=gen, sbj=emx.sbj))

    ## pick out common subjects
    sbj <- intersect(sbj, emx.sbj)
    gmx <- gmx[, sbj, , drop=F]
    hof <- lapply(hof, `[`, , sbj, drop=F)
    emx <- emx[, sbj, drop=F]
    list(gmx=gmx, hof=hof, emx=emx, pcs=pcs, sbj=sbj)
}

## sim('sim/GXP/01/000011873')
sim <- function
(
    fnm='sim/GXP/D04/01/000011873', ssz=NULL, efr=.05,
    wdp=NULL, ecp='gen', dst=c('gau', 'bin', 'poi'), r2=.5, ...)
{
    cat(fnm, ': ', sep='')
    dat = readSet(fnm)
    gmx = dat$gmx
    if(is.null(wdp))
        wdp <- length(dat$hof)
    hof = dat$hof[[wdp]]
    pcs = dat$pcs
    emx = dat$emx
    dot = list(...)
    
    ## mediators, drop PID, SEX, AGE, Education, Marriange
    phe <- readRDS('dat/GXP.rds')
    phe <- phe[, c("Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform")]
    sbj <- intersect(rownames(phe), dat$sbj)
    pmx <- t(scale(phe))                # use z-scores, column major
    
    ## common subjects
    gmx <- gmx[, sbj, , drop=F]
    hof <- hof[, sbj, drop=F]
    pcs <- pcs[, sbj, drop=F]
    emx <- emx[, sbj, drop=F]
    pmx <- pmx[, sbj, drop=F]

    ## dosage matrix
    dsg <- gmx[, , 1] + gmx[, , 2]
    dim(dsg) <- dim(gmx)[1:2]
    dsg <- impute(dsg)                  # imputation
    dsg <- rmDgr(dsg)                   # cleanup
    if(length(dsg) == 0)
    {
        cat('degenerated\n')
        return(NULL)
    }   
    ## search for true QTLs
    wDsg <- .wct(.hwu.GUS(t(dsg)))      # weight by dosage
    wHof <- .wct(.hwu.GUS(t(hof)))      # weight by high order features
    wPcs <- .wct(.hwu.GUS(t(pcs)))      # weight by high order features
    if('gen' %in% ecp)                  # gene expression
    {
        pvl.qtl.dsg <- hwu.dg2(t(emx), wDsg)
        pvl.qtl.hof <- hwu.dg2(t(emx), wHof)
        pvl.qtl.pcs <- hwu.dg2(t(emx), wPcs)
    }
    if('med' %in% ecp)                  # mediators
    {
        pvl.qtl.dsg <- hwu.dg2(t(pmx), wDsg)
        pvl.qtl.hof <- hwu.dg2(t(pmx), wHof)
        pvl.qtl.pcs <- hwu.dg2(t(pmx), wPcs)
    }
    
    ## * ---------- [ simulation ] ---------- *
    if(is.null(ssz))
        ssz <- ncol(dsg)
    idx <- sample.int(ssz) %% ncol(dsg) + 1
    dsg <- dsg[, idx, drop=F]
    hof <- hof[, idx, drop=F]
    pcs <- pcs[, idx, drop=F]
    emx <- emx[, idx, drop=F]
    pmx <- pmx[, idx, drop=F]

    ## * -------- [effect composition] -------- *
    x <- NULL
    if('gen' %in% ecp)                  # genes
        x <- rbind(x, emx)
    if('med' %in% ecp)
        x <- rbind(x, pmx)
    if('non' %in% ecp)
        x <- rbind(x, nmx)
    x <- t(scale(t(x)))
    nmv <- nrow(x)
    
    ## vcv <- tcrossprod(scale(t(x)))
    ## xb <- mvrnorm(1, rep(0, ncol(x)), vcv)
    
    .b <- rnorm(nrow(x))
    xb <- colSums(x * .b)
    if(r2 > 0.0)
        eta <- xb + rnorm(ssz, 0, sqrt(1/r2 - 1) * sd(xb))
    else
        eta <- runif(ssz, -sd(eta) +sd(eta))
    dst <- match.arg(dst, c('gau', 'bin', 'poi'))
    if(dst == 'poi')
        y <- rpois(ssz, exp(eta))
    else if(dst == 'bin')
        y <- rbinom(ssz, 1, 1/(1+exp(-eta)))
    else
        y <- eta

    ## * -------- U sta and P val -------- *
    wDsg <- .wct(.hwu.GUS(t(dsg)))
    wHof <- .wct(.hwu.GUS(t(hof)))
    wPcs <- .wct(.hwu.GUS(t(pcs)))
    pvl.sim.dsg <- hwu.dg2(y, wDsg)
    pvl.sim.hof <- hwu.dg2(y, wHof)
    pvl.sim.pcs <- hwu.dg2(y, wPcs)

    .m <- sprintf(
        '%s %3.1e %3.1e %3.1e %3.1e %3.1e %3.1e',
        dst,
        pvl.qtl.dsg, pvl.qtl.hof, pvl.qtl.pcs,
        pvl.sim.dsg, pvl.sim.hof, pvl.sim.pcs)
    cat(.m, '\n')

    nhv <- nrow(hof)
    ngv <- nrow(gmx)
    rec <- .record()
    rec
}

main <- function(fdr='sim/GXP/D06/01', out=NULL, rep=10, ...)
{
    fns <- dir(fdr, '*.pcs', full.names=T)
    fns <- sub('[.].*$', '', fns)
    
    fns <- sample(fns, rep, T)
    ret <- lapply(fns, sim, ...)
    ret <- lol2tab(ret)
    
    if(!is.null(out))
    {
        out <- sub('[.].*$', '.rds', out)
	saveRDS(ret, out)
    }
    invisible(ret)
}
