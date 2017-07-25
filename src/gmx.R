source('src/dsg.R')
source('src/utl.R')
source('src/hlp.R')
source('src/hwu.R')
source('src/gsm.R')
source('src/tgz.R')

## sim('sim/GXP/01/000011873')
sim <- function(fnm, ssz=NULL, mdl=~g, fam=NULL, r2=.05, pop=NULL, ...)
{
    ## fetch necessary data & infomation
    cat(fnm, ': ', sep='')
    dat <- readTGZ(fnm)
    gmx <- dat$gmx
    hof <- dat$hof
    rsd <- dat$rsd
    eot <- as.numeric(dat$inf$eot)
    nhf <- dat$nhf

    ## data matrics
    dsg <- gmx[, , 1] + gmx[, , 2]
    dim(dsg) <- dim(gmx)[1:2]
    dsg <- impute(dsg)                  # imputation
    dsg <- rmDgr(dsg)                   # cleanup
    if(length(dsg) == 0)
    {
        cat('degenerated\n')
        return(NULL)
    }
    xmx <- rbind((dsg > 0) * 1, (dsg > 1) * 1)
    
    ## * ---------- [ simulation ] ---------- *
    if(is.null(ssz))
        ssz <- ncol(dsg)
    idx <- sample.int(ncol(dsg), ssz)
    dsg <- dsg[, idx, drop=F]
    hof <- hof[, idx, drop=F]
    rsd <- rsd[, idx, drop=F]
    xmx <- xmx[, idx, drop=F]

    ## link function, distribution family
    fam <- substitute(fam)
    fam <- if(is.null(fam)) gau else fam
    fam <- eval(fam, .families)

    ## multiple populations
    .pp <- if(is.null(pop)) 1 else pop
    if(length(.pp) == 1)
        .pp <- rep(1, .pp)
    .i <- cbind(1:ssz, sample.int(length(.pp), ssz, T, .pp))
    xb <- sapply(1:length(.pp), function(.)
    {
        x <- gem(mdl, dsg, ...)         # terms
        b <- rnorm(ncol(x))             # coefs
        xb <- x %*% b                   # effects
        drop(xb)
    })[.i]
    y <- fam$ign(xb, r2=r2)
    
    ## * -------- U sta and P val -------- *
    wgt.dsg <- .wct(.hwu.GUS(t(dsg)))
    wgt.xmx <- .wct(.hwu.GUS(t(xmx)))
    wgt.ibs <- .wct(.hwu.IBS(t(dsg)))
    wgt.hof <- .wct(.hwu.GUS(t(hof)))
    wgt.rsd <- .wct(.hwu.GUS(t(rsd)))

    pvl.dsg <- hwu.dg2(y, wgt.dsg)
    pvl.hof <- hwu.dg2(y, wgt.hof)
    pvl.rsd <- hwu.dg2(y, wgt.rsd)
    pvl.xmx <- hwu.dg2(y, wgt.xmx)
    pvl.ibs <- hwu.dg2(y, wgt.ibs)
    
    fam <- fam$tag
    if(is.null(pop))
        rm(pop)
    else
        pop <- paste(pop, collapse=',')
    .m <- sprintf(
        '%s %3.1e %3.1e %3.1e %3.1e', fam,
        pvl.dsg, pvl.hof, pvl.rsd, pvl.xmx)
    cat(.m, '\n')

    mdl <- mdl.str(mdl)
    ngv <- nrow(gmx)
    rec <- .record(...)
    rec
}

main <- function(fr='sim/GMX/TE2', out=NULL, r2=.05, mdl=~g, rep=10, ...)
{
    if(file.exists(fr) && file.info(fr)$isdir)
        fns <- dir(fr, '*.tgz', full.names=T)
    else
        fns <- fr

    ## correct file suffix
    fns[file.exists(paste0(fns, '.tgz'))] <- paste0(fns, '.tgz')
    fns[file.exists(paste0(fns, '.tar.gz'))] <- paste0(fns, '.tar.gz')
    fns <- sample(fns, rep, T)
    ret <- lapply(fns, sim, r2=r2, mdl=mdl, ...)
    ret <- lol2tab(ret)
    
    if(!is.null(out))
    {
        out <- sub('[.].*$', '.rds', out)
	saveRDS(ret, out)
    }
    invisible(ret)
}
