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
    
    ## genomic map
    gmp <- readLines(file.path(tmpd, 'gmp.txt'))
    gmp <- sub('\\\\', '\t', gmp)
    gmp <- read.table(text=gmp, sep='\t')
    names(gmp) <- c('CHR', 'POS', 'ID')
    gvr <- gmp$ID
    ngv <- length(gvr)

    ## subjects
    sbj <- scan(file.path(tmpd, 'sbj.txt'), '', quiet=T)
    nsb <- length(sbj)
    ## untyped variants and subjects
    ## ugv <- scan(file.path(tmpd, 'ugv.txt'), 0L, quiet=T)
    ## usb <- scan(file.path(tmpd, 'usb.txt'), 0L, quiet=T)
    
    ## genomic matrix in 2-copy format
    gx0 <- scan(file.path(tmpd, 'gx0.txt'), what=0L, quiet=T)
    gx1 <- scan(file.path(tmpd, 'gx1.txt'), what=0L, quiet=T)
    gmx <- array(c(gx0, gx1), c(ngv, nsb, 2L),
                 list(gvr=gvr, sbj=sbj, cpy=c('CPY0', 'CPY1')))
    
    hof <- scan(file.path(tmpd, 'hof.txt'), what=.0, quiet=T)
    nhf <- length(hof) / nsb
    hof <- matrix(hof, nhf, nsb, F,
                  list(ftr=sprintf('H%03X', 1:nhf), sbj=sbj))

    list(gmx=gmx, hof=hof, gmp=gmp, inf=inf, sbj=sbj,
         eot=double(inf$eot),
         eov=double(inf$eov))
}

sim <- function(gmx, hof, ssz=100, efr=.05, r2=.05, ...)
{
    ## dots
    dot <- list(...)
    seq <- dot$seq
    eot <- dot$eot
    eov <- dot$eov
    
    ## dosage matrix
    dsg <- gmx[, , 1] + gmx[, , 2]
    dsg <- impute(dsg)                  # imputation

    ## pick some subjects
    idx <- sample.int(ncol(dsg), ssz, F)
    dsg <- dsg[, idx]
    hof <- hof[, idx]
    
    ## check genotype degeneration
    dsg <- rmDgr(dsg)

    ## * -------- [effects from typed variants] -------- *
    if('eft' %in% dot && dot[['eft']] == 'non')
    {
        ## dummy effect for type I error check
        .rs <- sqrt(sqrt(1 - r2) / r2)
        eta <- rnorm(ssz) + rnorm(ssz, 0, .rs)
    }
    else
    {
        .n <- nrow(dsg)                 # No. of variants
        .a <- dsg                       # additive
        .d <- 2 * (dsg > 0)             # dominence
        .r <- 2 * (dsg > 1)             # resessive
        .g <- rbind(.a, .d, .r)[1:.n + .n * sample(0:2, .n, T), ]
        
        ## .i <- combn(.n, 2)
        ## .i <- .g[.i[1,],] * .g[.i[2,],]
        ## .x <- rbind(.g, .i)
        .x <- .g
        .xb <- rnorm(nrow(.x)) * rbinom(nrow(.x), 1, efr) * .x
        .xb <- colSums(.xb)
        .rs <- sqrt((1 - r2) / r2 * var(.xb))
        eta <- .xb + rnorm(ssz, 0, .rs)
    }
    ## genome effect is linear
    y <- eta

    ## * -------- U sta and P val --------*
    pvl.dsg <- hwu.dg2(y, .wct(.hwu.GUS(t(dsg))))
    pvl.hof <- hwu.dg2(y, .wct(.hwu.GUS(t(hof))))
    nhv <- nrow(hof)

    rec <- .record()
    rec
}

main <- function(fns='sim/W09/SSS_EX4', out=NULL, n.i=100L, ...)
{
    fns <- dir(fns, '*.tgz', full.names=T)
    idx <- sample.int(length(fns), size = n.i, replace=T)
    fns <- fns[idx]
    seq <- sub('[.][^.]*$', '', basename(fns))
    
    ## get extra arguments, also remove NULL arguments so they won't
    ## cause havoc in expand.grid
    dot <- list(...)
    dot <- dot[!sapply(dot, is.null)]
    
    if(length(dot) < 1L)
        args <- data.frame(seq=seq, gnf=fns, stringsAsFactors = F)
    else
    {
        ## expands the repetitions.
        args <- list(idx = 1L:n.i, KEEP.OUT.ATTRS = F, stringsAsFactors = F)
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
    cat(paste(names(args[[1]]), collapse='\t'), '\n')
    rpt <- lapply(args, function(a)
    {
        cat(paste(a, collapse='\t'), '\n')
        a <- c(a, readTGZ(a$gnf))
        r <- do.call(sim, a)
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
