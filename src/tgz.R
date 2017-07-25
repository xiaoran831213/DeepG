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
    ## sbj <- scan(file.path(tmp, seq, 'sbj'), '', quiet=T)
    ## nsb <- length(sbj)
    
    ## genomic matrix, truth
    gx0 <- scan(file.path(tmp, seq, 'gx0'), 0L, quiet=T)
    gx1 <- scan(file.path(tmp, seq, 'gx1'), 0L, quiet=T)
    nsb <- length(gx0) / nrow(gmp)
    sbj <- sprintf('%03X', 1:nsb)
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

merge <- function(fs)
{
    ret <- list()
    fs <- lapply(fs, readTGZ)
    names(fs) <- paste0('dp', sapply(fs, function(f) f$inf$wdp))
    
    ## members shared across depth
    ret <- with(fs[[1]], list(seq=seq, gmx=gmx, gmp=gmp, sbj=sbj))
    
    ## members unique for each depth
    ret$enc <- lapply(fs, with, list(hof=hof, rsd=rsd, nhf=nhf))

    ret
}
