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
    w <- matrix(rnorm(D*D), D, D)
    w <- .3* (w + t(w))
    diag(w) <- 0
    s <- matrix(rbinom(length(N * D), 1, .5), N, D)
    s[, +1] <- 1
    s <- go(s, w, it=5)
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

test <- function(n=100, i=100, P=3, Q=3, ...)
{
    dt <- with(simu(n, P, Q*2), s[, v])
    LM <- glm(dt[, P] ~ dt[, 1:(P-1)], 'binomial')
    bm <- init(x=dt, q=max(1, Q/2))

    ## bm <- within(bm, s[, P] <- as.integer(s[, 1] + s[, 2]))
    s2 <- bm$s
    s2[, P+1] <- 0
    p2 <- with(bm, go(s2, w, up=c(P+1, bm$h), it=20))
    p3 <- prob(p2, bm$w, P+1)
    e2 <- sum(abs(bm$s[, P+1] - p2[, P+1]))
    e3 <- sum(bm$s[, P+1] * log(p3) + (1 - bm$s[, P+1]) * log(1-p3))
    cat(e2, "\t", e3, "\n")
    while(i > 0)
    {
        bm <- within(bm, w <- trn(s, w, h, it=50, mf=F, ...))
        p2 <- with(bm, go(s2, w, up=c(P+1, bm$h), it=20))
        p3 <- prob(p2, bm$w, P+1)
        e2 <- sum(abs(bm$s[, P+1] - p2[, P+1]))
        e3 <- sum(bm$s[, P+1] * log(p3) + (1 - bm$s[, P+1]) * log(1-p3))
        cat(e2, "\t", e3, "\n")
        i <- i - 1
    }

    ## GLM of visible units
    print(summary(LM))
    invisible(bm)
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
go <- function(s, w, up=NULL, it=1, mf=F)
{
    ## fixed units
    if(is.null(up))
        up <- 2:ncol(s)
    N <- nrow(s)

    while(it > 0)
    {
        if(mf)                      # 2.1 naive mean field
        {
            for(k in up)
            {
                s[, k] <- prob(s, w, k)
            }
        }
        else                        # 2.2 sampling instead.
        {
            for(k in up)
            {
                s[, k] <- rbinom(N, 1, prob(s, w, k))
            }
        }
        it <- it - 1
    }
    s
}


trn <- function(s, w, h, i=1, lrt=1e-3, ...)
{
    while(i > 0)
    {
        ## positive phase
        pp <- crossprod(go(s, w, up=h, ...))

        ## negative phase
        np <- crossprod(go(s, w, ...))
        
        ## update rule
        up <- pp - np
        diag(up) <- 0

        w <- w + lrt * up
        i <- i - 1
    }
    w
}
