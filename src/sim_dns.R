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
    ugv <- scan(file.path(tmpd, 'ugv.txt'), 0L, quiet=T)
    usb <- scan(file.path(tmpd, 'usb.txt'), 0L, quiet=T)
    
    ## genomic matrix
    ## dosage format
    dsg <- readLines(file.path(tmpd, 'dsg.txt'))
    stopifnot(nsb == length(dsg))
    dsg <- scan(text=dsg, what=0L, quiet=T)
    dsg <- matrix(dsg, ngv, nsb, dimnames=list(gvr=gvr, sbj=sbj))
    ## 2-copy format
    gx0 <- scan(file.path(tmpd, 'gx0.txt'), what=0L, quiet=T)
    gx1 <- scan(file.path(tmpd, 'gx1.txt'), what=0L, quiet=T)
    gmx <- array(
        c(gx0, gx1),
        c(ngv, nsb, 2L), list(gvr=gvr, sbj=sbj, cpy=c('CPY0', 'CPY1')))
    
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

    ## the network
    ## nwk <- list()
    ## for(i in 1:inf$ae1)
    ## {
    ##     nwi <- list()
    ##     for(f in paste(c('ew', 'dw'), i-1, '.txt', sep=''))
    ##     {
    ##         nt <- readLines(file.path(tmpd, f))
    ##         d1 <- length(nt)
    ##         nt <- scan(text=nt, what=.0, quiet=T)
    ##         d2 <- length(nt) / d1
    ##         nwi[[sub('[0-9]*.txt$', '', f)]] <- matrix(nt, d1, d2, T)
    ##     }
    ##     for(f in paste(c('eb', 'db'), i-1, '.txt', sep=''))
    ##     {
    ##         nwi[[sub('[0-9]*.txt$', '', f)]] <- scan(file.path(tmpd, f), .0, quiet=T)
    ##     }
    ##     nwk[[paste('d', i-1, sep='')]] <- nwi
    ## }

    list(gmx=gmx, hof=hof, gmp=gmp, inf=inf, dsg=dsg,
         usb=usb, ugv=ugv, sbj=sbj, etb=etb)
}

encode <- function(gmx, nwk)
{
    if(length(dim(gmx)) > 2L)
        hfi <- rbind(gmx[, , 1], gmx[, , 2])
    else
        hfi <- gmx
    for(i in 1:length(nwk))
    {
        nt <- nwk[[paste('d', i-1, sep='')]]
        hfi <- crossprod(nt$ew, hfi)
        hfi <- hfi + nt$eb
        hfi <- 1/(1+exp(-hfi))
    }
    hfi
}

enoise <- function(gmx, dns=.1)
{
    i <- rbinom(length(gmx), 1, dns)
    gmx[i > 0] <- 0L
    gmx
}

sim <- function(pck, ssz=100, eft=c('gno', 'non'), efr=.15, r2=.05, dns=NULL, ...)
{
    wdp <- as.integer(pck$inf$ae1)      # network depth
    eot <- pck$etb[wdp+1, 'eot']        # error of training
    eov <- pck$etb[wdp+1, 'eov']        # error of validation
    usb <- pck$usb > 0                  # untyped subjects
    ugv <- pck$ugv > 0                  # untyped variants
    gmx <- pck$gmx
    dsg <- pck$dsg
    nwk <- pck$nwk
    if(is.null(dns))
        dns <- as.numeric(pck$inf$dns)

    ## masks
    ## dosage matrix
    dsg <- impute(dsg)                # imputation
    dsg <- dsg[!ugv, !usb]            # typed variants, typed subjects
    gmx <- gmx[!ugv, !usb, ]

    ## pick some subjects
    idx <- sample.int(ncol(dsg), ssz, T)
    dsg <- dsg[, idx]
    gmx <- gmx[, idx, ]
    
    ## check genotype degeneration
    dsg <- rmDgr(dsg)

    ## * -------- [effects from typed variants] -------- *
    eft <- match.arg(eft, c('gno', 'non'))
    if(eft == 'non')
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
        .g <- rbind(.a, .d, .r)[1:.n + .n * sample(0L:2L, .n, T), ]

        ## .i <- combn(.n, 2)
        ## .i <- .g[.i[1,],] * .g[.i[2,],]

        ## .x <- rbind(.g, .i)
        .x <- dsg
        
        ## heterogenious effect
        .e1 <- rnorm(nrow(.x), +1) * rbinom(nrow(.x), 1, efr) * .x
        .e2 <- rnorm(nrow(.x), -1) * rbinom(nrow(.x), 1, efr) * .x
        .mx <- rbinom(nrow(.x), 1, .5)
        .xb <- rbind(.e1[.mx > 0, ], .e2[.mx < 1, ])
        .xb <- colSums(.xb)
        .rs <- sqrt((1 - r2) / r2 * var(.xb))
        eta <- .xb + rnorm(ssz, 0, .rs)
    }
    ## genome effect is linear
    ## y <- I(eta)
    y <- eta # * (abs(eta) > sd(eta))

    ## * --------  enable noises -------- *
    nmx <- enoise(gmx, dns)
    dsg <- nmx[, , 1] + nmx[, , 2]
    nmx <- rbind(nmx[, , 1], nmx[, , 2])

    ## * ------ high order feature ------ *
    hof <- encode(nmx, nwk)
    
    ## * -------- U sta and P val --------*
    pvl.gs1 <- hwu.dg2(y, .wct(.hwu.GUS(t(nmx))))
    pvl.gs2 <- hwu.dg2(y, .wct(.hwu.GUS(t(dsg))))
    pvl.hof <- hwu.dg2(y, .wct(.hwu.GUS(t(hof))))

    rec <- .record()
    rec
}

main <- function(fns='sim/W09/30_FT5', out=NULL, n.i=100L, ...)
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
    rpt <- lapply(args, function(a)
    {
        ## print(a$gnf)
        cat(paste(a, collapse='\t'), '\n')
        ## r <- with(a, sim(pck, ssz, eft, efr, r2, wdp))
        a$pck <- readTGZ(a$gnf)
        r <- do.call(sim, a)
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
