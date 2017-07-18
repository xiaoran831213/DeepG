## Boltzmann Machine
library(MASS)

## simulate an experimental data
## n: sample size, 
## p: true number of visible units
## q: true number of hidden units
simu <- function(n=500, p=3, q=5)
{
    ## dimension
    d <- p + q

    ## mean and variance-covariance of all units in the logit space
    cv <- tcrossprod(matrix(rnorm(d*d, sd=5), d, d))
    
    ## signal in logit space
    mu <- rnorm(d)
    et <- mvrnorm(n, mu, cv)

    ## system states in data space
    s <- rbinom(length(et), 1, 1/(1+exp(-et)))
    dim(s) <- dim(et)

    list(s=s, v=1:p, h=(p+1):(p+q), p=p, q=q, cv=cv, mu=mu)
}

## initialize BM according to problem data
## x: the visible data
## q: number of hidden units
init <- function(x, q=3)
{
    n <- nrow(x)
    p <- ncol(x)
    l <- c('b', paste0('v', 1:p), paste0('h', 1:q))

    ## prepend bias inducer, append initial hidden units
    s <- cbind(1, x, matrix(rbinom(n * q, 1, .5), n, q))
    colnames(s) <- l
    d <- ncol(s)

    ## initialize connections
    w <- matrix(rnorm(d*d), d, d, dimnames=list(l, l))
    w[upper.tri(w)] <- t(w)[upper.tri(w)]
    diag(w) <- 0

    list(s=s, v=2:(p+1), h=(p+2):(p+1+q), w=w)
}

test <- function(n=100, i=100)
{
    ## ev <- simu(600, 2, 3)
    x1 <- rbinom(n, 1, .5)
    x2 <- rbinom(n, 1, .5)
    x3 <- as.integer(x1 > x2)
    x <- cbind(x1=x1, x2=x2, x3=x3)
    bm <- init(x=x, q=5)
    s2 <- bm$s
    s2[, 4] <- 0
    er <- sum(abs(bm$s[, 4] - s2[, 4]))
    print(er)
    while(i > 0)
    {
        bm <- within(bm, w <- update(s, w, h, lrt=1e-2, mf=T))
        p2 <- go(s2, bm$w, i=10, m=c(4, bm$h))
        er <- sum(abs(bm$s[, 4] - p2[, 4]))
        print(er)
        i <- i - 1
    }
    invisible(bm)
}

## system energy from states and connections
E <- function(s, w)
{
    - 0.5 * rowSums(s %*% w * s)
}

pr <- function(s, w, m=NULL)
{
    ## Pr(s_ik == 1) = sigmoid( sum_j^d{s_ij w_jk} )
    if(is.null(m))
        m <- -1
    1 / (1 + exp(-s %*% w[, m]))
}

## advance the system states
go <- function(s, w, i=1, m=NULL, mf=F)
{
    if(is.null(m))
        m <- -1
    while(i > 0)
    {
        p <- pr(s, w, m)
        if(mf)
            s[, m] <- p
        else
            s[, m] <- rbinom(length(p), 1, p)
        i <- i - 1
    }
    s
}


update <- function(s, w, h, i=1, lrt=1e-3, ...)
{
    while(i > 0)
    {
        ## positive phase
        pp <- crossprod(go(s, w, 10, h, ...))

        ## negative phase
        np <- crossprod(go(s, w, 10, ...))
        
        ## update rule
        up <- pp - np
        diag(up) <- 0

        w <- w + lrt * up
        i <- i - 1
    }
    ## print(up)
    w
}
