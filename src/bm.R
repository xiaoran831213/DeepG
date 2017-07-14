## Boltzmann Machine
library(MASS)

## generate a dxd symmetric positive definite matrix
.spd <- function(d=5, r=.632)
{
    ## draw a random matrix
    A <- matrix(rchisq(d * d, 1) * rbinom(d * d, 1, r), d, d)
    
    ## make it symmetric
    A <- tcrossprod(A)

    
    A
}

simu <- function(n=500, p=3, q=5, pc=.8)
{
    ## mean and variance-covariance of all units in the logit space
    vc <- .spd(r=pc)
    mu <- rnorm(p+q)

    ## draw state values in logit space
    s <- mvrnorm(n, mu, vc)
    s
}

E <- function(v=NULL, h=NULL, s=NULL, b=NULL, w=NULL)
{
    ## the collection of all units
    s <- cbind(v, h, s)
    p <- ifelse(is.null(v), 0, NCOL(v))
    q <- ifelse(is.null(p), 0, NCOL(h))

    ## offsets and graph
    if(is.null(b))
        b <- rep(0, p+q)
    if(is.null(w))
        w <- matrix(0, p+q, p+q)

    ## energy for all samples
    ## Sb + SWS'
    s * b + tcrossprod(s %*% w, s)
}
