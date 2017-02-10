source('src/dsg.R')
source('src/utl.R')
source('src/hlp.R')
source('src/hwu.R')

readTGZ <- function(tgz)
{
    tmpd <- tempdir()
    untar(tgz, exdir=tmpd)
    data <- list()

    ## meta information
    inf <- readLines(file.path(tmpd, 'inf.txt'))
    inf <- strsplit(inf, '=')
    nms <- sapply(inf, `[`, 1L)
    inf <- lapply(inf, `[`, 2L)
    names(inf) <- nms
    inf <- inf[c('ae1', 'cvk', 'cvl', 'lrt', 'fnm', 'out', 'nwk')]

    ## genomic map
    gmp <- readLines(file.path(tmpd, 'gmp.txt'))
    gmp <- sub('\\\\', '\t', gmp)
    gmp <- read.table(text=gmp, sep='\t')
    names(gmp) <- c('CHR', 'POS', 'ID')
    gvr <- gmp$ID
    ngv <- length(gvr)

    ## subjects
    ## sbj <- scan(file.path(tmpd, 'sbj.txt'), '', quiet=T)
    
    ## genomic matrix
    gmx <- readLines(file.path(tmpd, 'dsg.txt'))
    nsb <- length(gmx)
    sbj <- sprintf('%04X', 1:nsb)
    gmx <- scan(text=gmx, what=0L, quiet=T)
    gmx <- matrix(gmx, ngv, nsb, dimnames=list(gvr=gvr, sbj=sbj))

    ## untyped variants
    ## umx <- scan(file.path(tmpd, 'usg.txt'), 0L, quiet=T)
    ## umx <- matrix(
    ##     umx, length(umx) / length(sbj), length(sbj))

    ## the error table
    etb <- readLines(file.path(tmpd, 'etb.txt'))
    etb <- scan(text=etb, what=.0, quiet=T)
    ndp <- length(etb)/2
    dnm <- list(paste('d', 0:(ndp-1), sep=''), c('eot', 'eov'))
    etb <- matrix(etb, ndp, 2, byrow=T, dimnames=dnm)
    
    ## high order features
    hof <- list()
    for(i in 0:9)
    {
        f <- file.path(tmpd, paste('hf', i, '.txt', sep=''))
        if(!file.exists(f))
            next
        h <- scan(f, .0, quiet=T)
        r <- length(h)/nsb
        dnm <- list(ftr=sprintf('H%03X', 1:r), sbj=sbj)
        h <- matrix(h, r, nsb, dimnames=dnm)
        hof[[paste('d', i, sep='')]] <- h
    }
    list(gmx=gmx, hof=hof, gmp=gmp, inf=inf, sbj=sbj, etb=etb)
}

.sim <- function(pck, ssz=100,
                 eft=c('gno', 'non'),
                 efr=.15, r2=.10, wdp=3, ...)
{
    .dp <- paste('d', wdp, sep='')
    eot <- pck$etb[.dp, 'eot']
    eov <- pck$etb[.dp, 'eov']
    hof <- pck$hof[[.dp]]
    
    ## genomic matrix
    gmx <- pck$gmx
    ssp <- ncol(gmx)                    # size of sample pool
    gmx <- impute(gmx)

    ## pick some subjects
    idx <- sample.int(ncol(gmx), ssz, T)
    ## umx <- umx[, idx]
    gmx <- gmx[, idx]
    hof <- hof[, idx]
    
    ## check genotype degeneration
    gmx <- rmDgr(gmx)
    ## umx <- rmDgr(umx)
    ngv <- nrow(gmx)                   # number of genomic variants
    nhf <- nrow(hof)                   # number of high order features

    ## * -------- [effects from untyped variants] -------- *
    eft <- match.arg(eft, c('gno', 'non'))
    if(eft == 'non')
    {
        ## dummy effect for type I error check
        .rs <- sqrt(sqrt(1 - r2) / r2)
        eta <- rnorm(ssz) + rnorm(ssz, 0, .rs)
    }
    else
    {
        ## additive effect
        .xb <- rnorm(nrow(gmx)) * rbinom(nrow(gmx), 1, efr) * gmx
        ## .xb <- rnorm(nrow(umx)) * umx
        .xb <- colSums(.xb)
        .rs <- sqrt((1 - r2) / r2 * var(.xb))
        eta <- .xb + rnorm(ssz, 0, .rs)
    }
    ## genome effect is linear
    y <- I(eta)
    
    ## * -------- U sta and P val --------*
    pvl.gno <- hwu.dg2(y, .wct(.hwu.IBS(t(gmx))))
    pvl.hof <- hwu.dg2(y, .wct(.hwu.GUS(t(hof))))

    rec <- .record()
    rec
}

main <- function(fns='sim/W08/30_HW5', fn2='sim/W08/31_HW5', out=NULL, n.i=100L, ...)
{
    fns <- dir(fns, '*.tgz', full.names=T)
    fn2 <- dir(fn2, '*.tgz', full.names=T)
    
    idx <- sample.int(length(fns), size = n.i, replace=T)
    fns <- c(fns[idx], fn2[idx])
    seq <- sub('[.][^.]*$', '', basename(fns))
    
    ## get extra arguments, also remove NULL arguments so they won't
    ## cause havoc in expand.grid
    dot <- list(...)
    dot <- dot[!sapply(dot, is.null)]
    
    if(length(dot) < 1L)
        args <- data.frame(seq=seq, gnf=fns, stringsAsFactors = F)
    else
    {
        ## idx = 1L:n.i expands the repetitions.
        args <- list(idx = 1L:(2*n.i), KEEP.OUT.ATTRS = F, stringsAsFactors = F)
        args <- do.call(expand.grid, c(dot, args))
        args <- args[do.call(order, args[, names(dot), drop = F]), ]
        args <- within(
            args,
        {
            gnf <- fns[idx]
            seq <- seq[idx]
            rm(idx)
        })
    }
    
    ## turn data.frame to a list of lists
    args <- tab2lol(args)
    
    ## repeatative simulation
    rpt <- lapply(args, function(a)
    {
        ## print(a$gnf)
        cat(paste(a, collapse='\t'), '\n')
        a$pck <- readTGZ(a$gnf)
        r <- do.call(.sim, a)
        r$seq <- a$seq
        r
    })
    ## rpt <- do.call(rbind, rpt)
    rpt <- lol2tab(rpt)

    ## report and return
    if(!is.null(out))
    {
        if(!grepl('[.]rds$', out))
           out <- paste(out, 'rds', sep='.')
        saveRDS(rpt, out)
    }
    invisible(rpt)
}
