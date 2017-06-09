## link functions

## gaussian
gau <- within(list(),
{
    tag <- 'gau'
    lnk <- function(mu)        mu
    inv <- function(eta)       eta
    gen <- function(mu,  sd=1) rnorm(length(mu),  mu,  sd)
    ign <- function(eta, sd=1) rnorm(length(eta), eta, sd)
})

## binomial
bin <- within(list(),
{
    tag <- 'bin'
    lnk <- function(mu)           log(mu / (1 - mu))
    inv <- function(eta)          1 / (1 + exp(-eta))
    gen <- function(mu,  size=1)  rbinom(length(mu),  size, mu)
    ign <- function(eta, size=1)  rbinom(length(eta), size, inv(eta))
})

## poisson
poi <- within(list(),
{
    tag <- 'poi'
    lnk <- function(mu)  log(mu)
    inv <- function(eta) exp(eta)
    gen <- function(mu)  rpois(length(mu), mu)
    ign <- function(eta) rpois(length(eta), exp(eta))
})

## sin
snl <- within(list(),
{
    tag <- 'sin'
    lnk <- function(mu)  sin(mu)
    inv <- function(eta) cos(eta)
    gen <- function(mu)  sin(mu)
    ign <- function(eta) cos(eta)
})
.families <- list(gau=gau, bin=bin, poi=poi, sin=snl)

## cos
csl <- within(list(),
{
    tag <- 'sin'
    lnk <- function(mu)  cos(mu)
    inv <- function(eta) sin(eta)
    gen <- function(mu)  cos(mu)
    ign <- function(eta) sin(eta)
})
.families <- list(gau=gau, bin=bin, poi=poi, sin=snl, cos=csl)


## genetic effect model
gem <- function(mdl=~ a + d + r, G, gcd=1, max.gvr=32, max.tms=4096, rm.nic=T, ...)
{
    ## pick genomic variables
    if(nrow(G) > max.gvr)
        G <- G[sort(sample.int(nrow(G), max.gvr)), ]
    
    ## terms and varialbes
    tms <- terms(mdl)
    attr(tms, 'intercept') <- 0
    attr(tms, 'response') <- 0
    vs <- all.vars(tms)

    ## at least 3 variants are needed to assign 3 basic genetic effect
    if(nrow(G) < 3L)
        G <- rbind(G, matrix(0L, 3L - nrow(G), ncol(G)))
    
    ## genetic bases
    gbs <- list()
    if('a' %in% vs || 'g' %in% vs)      # additive
        gbs[['a']] <- G                 # - 1
    if('d' %in% vs || 'g' %in% vs)      # dominance
        gbs[['d']] <- (G>0) * 2         # - 1
    if('r' %in% vs || 'g' %in% vs)      # rescessive
        gbs[['r']] <- (G>1) * 2         # - 1
    if(gcd > 0)
        gbs <- lapply(gbs, `-`, 1)
    
    ## evenly assign the 3 type of bases to variants
    msk <- sample(1:nrow(G) %% length(gbs) + 1)
    for(i in 1:length(gbs))
    {
        gbs[[i]] <- gbs[[i]][msk==i, , drop=F]
    }
    gbs[['g']] <- do.call(rbind, gbs)
    gbs <- lapply(gbs, t)

    ## prepare model matrix
    mm <- model.matrix(tms, gbs)
    rownames(mm) <- colnames(G)
    colnames(mm) <- gsub('[I( )]', '', colnames(mm))
    assign <- attr(mm, 'assign')
    
    ## keep informative columns
    msk <- apply(mm, 2, sd) != 0
    drp <- ncol(mm) - sum(msk)
    if(drp > 0 && rm.nic)
    {
        print(sprintf('drop %d non-informative columns.', drp))
        mm <- mm[, msk, drop=FALSE]
        assign <- assign[msk]
    }

    ## crop excessive columns
    drp <- ncol(mm) - max.tms
    if(drp > 0)
    {
        print(sprintf('drop %d excessive columns.', drp))
        idx <- sort(sample.int(ncol(mm), max.tms))
        mm <- mm[, idx]
        assign <- assign[idx]
    }
    attr(mm, 'assign') <- assign
    mm
}

test <- function()
{
    P <- 6
    g1 <- matrix(sample.int(3, P*9, T)-1, P, 9)
    rownames(g1) <- sprintf('%02d', 1:P)
    
    ## m1 <- gem(~ g:I(g) + I(-g^2), G=g1, rm.nic=F)
    m1 <- gem(~ exp(g), G=g1, rm.nic=F)
    m1
}
