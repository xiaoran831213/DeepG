## Boltzmann Machine
library(MASS)

## simulate an experimental data
## n: sample size, 
## p: true number of visible units
## q: true number of hidden units
simu <- function(N=500, P=3, Q=5)
{
    ## dimension
    D <- 1 + P + Q

    ## mean and variance-covariance of all units in the logit space
    ## cv <- tcrossprod(matrix(rnorm(D*D), D, D))
    
    ## signal in logit space
    ## et <- mvrnorm(N, rep(0, D), cv)
    

    ## system states in data space
    ## s <- rbinom(length(et), 1, 1/(1+exp(-et)))
    w <- matrix(rnorm(D*D) * rbinom(D*D, 1, .8), D, D)
    w <- .4 * (w + t(w))
    w[abs(w) < 1] <- 0
    
    diag(w) <- 0
    s <- matrix(rbinom(length(N * D), 1, .5), N, D)
    s[, +1] <- 1
    s <- go(s, w, brn=5)
    s <- s[, -1]
    
    ## feature names
    ft <- paste0('v', 1:P)
    if(Q > 0)
        ft <- c(ft, paste0('h', 1:Q))
    dimnames(s) <- list(sbj=sprintf('S%03X', 1:N), ftr=ft)

    list(s=s, v=1:P, h=(P+1):(P+Q), P=P, Q=Q)
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

ce2 <- function(p, y)
{
    sum(-2 * (y * log(p) + (1 - y) * log(1 - p)))
}


test <- function(n=100, i=100, P=3, Q=3, ...)
{
    ## history
    hs <- double(i)
    
    ## simulate data
    dt <- with(simu(n, P, Q*2), s[, v])

    ## GLM reference
    y <- dt[, P]
    x1 <- dt[, 1:(P-1)]
    x2 <- dt[, 1:(P-1)]
    LM <- glm(y ~ x1, 'binomial')
    
    ## the Boltzmann Machine
    BM <- init(x=dt, q=Q)

    ## a damaged data
    s2 <- BM$s
    s2[, P+1] <- rbinom(n, 1, .5)

    ## make initial assessment of deviance
    dv <- with(BM,
    {
        ## sampling the response, and hidden units
        gs <- go(s2, w, idx=c(P+1, h), brn=20)

        ## activation probability of the response
        pp <- prob(gs, w, P+1)

        ## prob v.s. truth -> deviance
        ce2(pp, s[, P+1])
    })

    ## print and return
    cat(dv, "\n")
    hs[i] <- dv

    while(i > 0)
    {
        ## train the BM
        BM <- within(BM, w <- trn(s, w, h, nep=1, brn=20, gsp=5 ,...))
        i <- i - 1

        ## assess model deviance
        dv <- with(BM,
        {
            ## sampling the response, and hidden units
            gs <- go(s2, w, idx=c(P+1, h), brn=20)

            ## activation probability of the response
            pp <- prob(gs, w, P+1)

            ## prob v.s. truth -> deviance
            ce2(pp, s[, P+1])
        })

        ## print and return
        cat(dv, "\n")
        hs[i] <- dv
    }

    ## GLM of visible units
    print(summary(LM))
    BM$hs <- rev(hs)
    invisible(BM)
}

## system energy from states and connections
E <- function(s, w)
{
    - 0.5 * rowSums(s %*% w * s)
}

prob <- function(s, w, mk=NULL)
{
    ## Pr(s_k == 1) = sigmoid( sum_j^d{s_j w_jk} )
    if(is.null(mk))
        1 / (1 + exp(-s %*% w))
    else
        1 / (1 + exp(-s %*% w[,mk]))
}

## advance the system states
go <- function(s, w, idx=NULL, brn=20, gsp=1, drop=TRUE)
{
    ## fixed units
    if(is.null(idx))
        idx <- 2:ncol(s)
    N <- nrow(s)

    ## burn in
    for(i in 1:brn)
    {
        for(k in idx)
            s[, k] <- rbinom(N, 1, prob(s, w, k))
    }

    ## sampling
    r <- array(.0, c(dim(s), gsp))
    for(i in 1:gsp)
    {
        for(k in idx)
            s[, k] <- rbinom(N, 1, prob(s, w, k))
        r[, , i] <- s
    }

    ## return
    if(drop)
        r <- drop(r)
    r
}


trn <- function(s, w, h, nep=1, lrt=1e-3, ...)
{
    d <- 2:ncol(s)
    for(i in 1:nep)
    {
        ## positive phase
        pp <- go(s, w, idx=h, drop=F, ...)
        pp <- apply(pp, 3, crossprod)        #dim: length(w) X gbs
        pp <- rowMeans(pp)

        ## negative phase
        np <- go(s, w, idx=d, drop=F, ...)
        np <- apply(np, 3, crossprod)
        np <- rowMeans(np)
        
        ## update rule
        up <- pp - np
        dim(up) <- dim(w)
        diag(up) <- 0

        w <- w + lrt * up
    }
    w
}
