## check is an object is a scalar
.scalar <- function(obj)
{
    if(is.null(obj))
        return(TRUE)
    if(is.list(obj))
        return(FALSE)
    if(is.vector(obj) & length(obj) < 2L)
        return(TRUE)
    if(is.factor(obj) & length(obj) < 2L)
        return(TRUE)
    FALSE
}

## collect object in a function environment, by default only
## visible scalars are collected
.record <- function(env=parent.frame(), pass=.scalar, rm.null=0, ...)
{
    ret <- list()
    for(nm in ls(env))
    {
        obj <- env[[nm]]
        if(!pass(obj))
            next
        if(is.null(obj))
           obj <- 'NL'
        ret[[nm]] <- obj
    }
    dot <- list(...)
    for(nm in ls(dot))
    {
        obj <- dot[[nm]]
        if(!pass(obj))
            next
        if(is.null(obj))
            obj <- 'NL'
        ret[[nm]] <- obj
    }
    ret
}

## rescale to [0, 1]
.sc1 <- function(x)
{
    max.x <- max(x)
    min.x <- min(x)
    (x - min.x) / (max.x - min.x)
}

## standardize x to mean 0 and sd 1
.std <- function(x, na.rm = F)
{
    x <- x - mean(x, na.rm = na.rm)
    s <- sd(x, na.rm = na.rm)
    if(is.na(s) || s ==0)
        x
    else
        x / s
}

.ns1 <- function(x, sd)
{
    ## add normal noise
    x + rnorm(length(x), sd = sd)
}
## lower triangle of a matrix
.lwt <- function(x, ret.idx=0L)
{
    n <- nrow(x)
    z <- sequence(n)
    idx <- cbind(
        i = unlist(lapply(2L:n, seq.int, to=n), use.names = FALSE),
        j = rep.int(z[-n], rev(z)[-1L]))
        
    if(ret.idx == 1L)
        with(idx, i * n + j)
    else if(ret.idx == 2L)
        idx
    else
        x[idx]
}

.rds.rpt <- function(src, ...)
{
    ## pick out images by file name
    dirs <- c(src, ...)
    
    fns <- unlist(lapply(dirs, function(d) file.path(d, dir(d, '*.rds'))))
    dat <- lapply(fns, readRDS)
    do.call(rbind, dat)
}

