source('src/dsg.R')
source('src/utl.R')
source('src/hlp.R')
source('src/hwu.R')

readTGZ <- function(tgz)
{
    tmpd <- tempdir()
    data <- c('dsg', 'usg', 'inf', 'gmp', 'hof')
    data <- paste(data, 'txt', sep='.')
    untar(tgz, data, exdir=tmpd)
    data <- list()

    ## meta information
    inf <- readLines(file.path(tmpd, 'inf.txt'))
    inf <- strsplit(inf, '=')
    nms <- sapply(inf, `[`, 1L)
    inf <- lapply(inf, `[`, 2L)
    names(inf) <- nms

    ## genomic map
    gmp <- readLines(file.path(tmpd, 'gmp.txt'))
    gmp <- sub('\\\\', '\t', gmp)
    gmp <- read.table(text=gmp, sep='\t')
    names(gmp) <- c('CHR', 'POS', 'ID')

    ## genomic matrix
    gmx <- readLines(file.path(tmpd, 'dsg.txt'))
    nsb <- length(gmx)
    gmx <- scan(text=gmx, what=0L, quiet=T)
    gmx <- matrix(gmx, length(gmx) / nsb, nsb)

    ## untyped variants
    umx <- scan(file.path(tmpd, 'usg.txt'), 0L, quiet=T)
    umx <- matrix(umx, length(umx) / nsb, nsb)
    
    ## high order features
    hof <- scan(file.path(tmpd, 'hof.txt'), .0, quiet=T)
    nhf <- length(hof) / nsb
    hof <- matrix(hof, nhf, nsb)
    
    list(gmx=gmx, umx=umx, hof=hof, gmp=gmp, inf=inf)
}

.sim <- function(pck, ssz=100, use=c('all', 'mbd', 'vld'),
                 eft=c('gno', 'non'),
                 efr=.15, r2=.05, ...)
{
    ## genomic matrix and high order features
    use <- match.arg(use, c('all', 'mbd', 'vld'))
    idx <- list(mbd=1L:1750L, vld=-1L:-1750L, all=-0x7FFF)[[use]]
    umx <- pck$umx[, idx]
    gmx <- pck$gmx[, idx]
    hof <- pck$hof[, idx]
    
    ## guess genomic NA
    gmx <- impute(gmx)

    ## pick some subjects
    idx <- sample.int(ncol(gmx), ssz, T)
    umx <- umx[, idx]
    gmx <- gmx[, idx]
    hof <- hof[, idx]
    
    ## check genotype degeneration
    gmx <- rmDgr(gmx)
    umx <- rmDgr(umx)
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
    .pvl.gno <- hwu.dg2(y, .wct(.hwu.IBS(t(gmx))))
    .pvl.hof <- hwu.dg2(y, .wct(.hwu.GUS(t(hof))))

    eov <- as.numeric(pck$inf$verr)     # error of validation
    eot <- as.numeric(pck$inf$terr)     # error of training
    lmd <- as.numeric(pck$inf$lmd)      # weight decay lambda

    rec <- .record()
    ret <- list(
        c(rec, knl='gno', pvl=.pvl.gno),
        c(rec, knl='hof', pvl=.pvl.hof))
    ret <- lol2tab(ret)
    ret
}

main <- function(fns='raw/W08/32_FTN', out=NULL, n.i=50L, ...)
{
    fns <- dir(fns, '*.tgz', full.names=T)
    fns <- sample(fns, size = n.i, replace=T)

    ## get extra arguments, also remove NULL arguments so they won't
    ## cause havoc in expand.grid
    dot <- list(...)
    dot <- dot[!sapply(dot, is.null)]
    
    if(length(dot) < 1L)
        args <- data.frame(gnf=fns, stringsAsFactors = F)
    else
    {
        ## idx = 1L:n.i expands the repetitions.
        args <- list(idx = 1L:n.i, KEEP.OUT.ATTRS = F, stringsAsFactors = F)
        args <- do.call(expand.grid, c(dot, args))
        args <- args[do.call(order, args[, names(dot), drop = F]), ]
        args <- within(
            args,
        {
            gnf <- fns[idx]
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
        do.call(.sim, a)
    })
    rpt <- do.call(rbind, rpt)

    ## report and return
    if(!is.null(out))
    {
        if(!grepl('[.]rds$', out))
           out <- paste(out, 'rds', sep='.')
        saveRDS(rpt, out)
    }
    invisible(rpt)
}
