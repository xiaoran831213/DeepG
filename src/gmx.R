source('src/dsg.R')
source('src/utl.R')
source('src/hlp.R')
source('src/hwu.R')
source('src/gsm.R')


readTGZ <- function(tgz)
{
    ## tmp <- tempfile('', 'tmp')
    tmp <- tempfile('')
    seq <- sub('[.][^.]*$', '', basename(tgz))
    untar(tgz, exdir=tmp)

    ## meta information
    inf <- readLines(file.path(tmp, seq, 'inf'))
    inf <- strsplit(inf, '=')
    nms <- sapply(inf, `[`, 1L)
    inf <- lapply(inf, `[`, 2L)
    names(inf) <- nms
    
    ## genomic map
    gmp <- readLines(file.path(tmp, seq, 'gmp'))
    gmp <- sub('\\\\', '\t', gmp)
    gmp <- read.table(text=gmp, sep='\t')
    names(gmp) <- c('CHR', 'POS', 'ID')
    gvr <- sub('^b', '', gmp$ID)
    gmp$ID <- gvr
    ngv <- length(gvr)

    ## subjects
    sbj <- scan(file.path(tmp, seq, 'sbj'), '', quiet=T)
    nsb <- length(sbj)
    
    ## genomic matrix, truth
    gx0 <- scan(file.path(tmp, seq, 'gx0'), 0L, quiet=T)
    gx1 <- scan(file.path(tmp, seq, 'gx1'), 0L, quiet=T)
    gmx <- array(
        c(gx0, gx1), c(ngv, nsb, 2L),
        list(gvr=gvr, sbj=sbj, cpy=c('CP0', 'CP1')))
    
    ## genomic matrix, residual
    rsd <- scan(file.path(tmp, seq, 'rsd'), .0, quiet=T)
    rsd <- matrix(rsd, ngv * 2, nsb)
    colnames(rsd) <- sbj

    ## high order features
    hof <- scan(file.path(tmp, seq, 'hof'), .0, quiet=T)
    nhf <- length(hof) / nsb
    hof <- matrix(hof, nhf, nsb)
    colnames(hof) <- sbj

    unlink(tmp, T, T)
    list(
        seq=seq, gmx=gmx, hof=hof, rsd=rsd,
        gmp=gmp, nhf=nhf, inf=inf, sbj=sbj)
}


## sim('sim/GXP/01/000011873')
sim <- function(
                fnm='0005.tgz',
                ssz=NULL,
                mdl=~g,
                fam=NULL,
                r2=.05,
                pop=NULL,
                ...)
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
