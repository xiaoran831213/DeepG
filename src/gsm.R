## link functions

## gaussian
gau <- within(list(),
{
    tag <- 'gau'
    lnk <- function(mu) mu
    inv <- function(et) et
    gen <- function(mu, sd=1, r2=1)
    {
        y <- rnorm(length(mu), mu, sd)
        if(r2 > 0)
            y <- y + rnorm(length(y), 0, sqrt(1/r2-1) * sd(y))
        else
            y <- rnorm(length(y), 0, sd(y))
        y
    }
    ign <- function(et, sd=1, r2=1)
    {
        gen(inv(et), sd, r2)
    }
})

## binomial
bin <- within(list(),
{
    tag <- 'bin'
    lnk <- function(mu) log(mu / (1 - mu))
    inv <- function(et) 1/ (1 + exp(-et))
    gen <- function(mu, size=1, r2=1)
    {
        mu <- (1-sqrt(r2)) * 0.5 + sqrt(r2) * mu
        rbinom(length(mu), size, mu)
    }
    ign <- function(et, size=1, r2=1)
    {
        gen(inv(et), size, r2)
    }
})

## poisson
poi <- within(list(),
{
    tag <- 'poi'
    lnk <- function(mu)  log(mu)
    inv <- function(et)  exp(et)
    gen <- function(mu)  rpois(length(mu), mu)
    ign <- function(et)  gen(inv(et))
})

## sin
snl <- within(list(),
{
    tag <- 'sin'
    lnk <- function(mu, sin.off=0, sin.frq=1)
    {
        (sin(mu) / pi - sin.off) / sin.frq
    }
    inv <- function(et, sin.off=0, sin.frq=1)
    {
        cos(pi * (sin.off + sin.frq * et))
    }
    gen <- function(mu, r2=1)
    {
        if(r2 > 0)
            mu + rnorm(length(mu), 0, sqrt(1/r2-1) * sd(mu))
        else
            rnorm(length(mu), 0, sd(mu))
    }
    ign <- function(et, r2=1, ...)
    {
        gen(inv(et, ...), r2)
    }
})
.families <- list(gau=gau, bin=bin, poi=poi, sin=snl)


## genetic effect model
gem <- function(mdl=~ a + d + r, G,   # model, and raw data.
                gcd=0,                # genotype coding
                max.gvr=32,           # maximum number of variants
                max.tms=4096,         # naximum number of terms
                rm.nic=T,             # remove non-informative columns
                rm.dup=T,             # remove unintentionally duplicated columns
                baf=F,                # effect by allele frequency?
                fpr=0,                # frequency of parity
                ...)
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
    if(any(c('a', 'g') %in% vs))
        gbs[['a']] <- G                 # additive
    if(any(c('d', 'g') %in% vs))
        gbs[['d']] <- (G>0) * 2         # dominance
    if(any(c('r', 'g') %in% vs))
        gbs[['r']] <- (G>1) * 2         # rescessive
    if(gcd > 0)
        gbs <- lapply(gbs, `-`, 1)


    ## by allele frequency (favor low MAF)?
    maf <- rowMeans(G) / 2.0
    if(baf != 0)
    {
        gbs <- lapply(gbs, function(g)
        {
            if(baf > 0)
                r <- g * rnorm(length(g), 0, (1 -maf) * baf)
            else
                r <- g * rnorm(length(g), 0, maf * -baf)
        })
    }

    ## evenly assign the 3 type of bases to variants
    msk <- sample(1:nrow(G) %% length(gbs) + 1)
    for(i in 1:length(gbs))
    {
        gbs[[i]] <- gbs[[i]][msk==i, , drop=F]
    }
    gbs[['g']] <- do.call(rbind, gbs)

    ## parity, pooled from 4 differenct sets of variants
    if('p' %in% vs && fpr > 0)
    {
        ## number of parity-variants
        npr <- ifelse(fpr > 1, fpr, nrow(G) * fpr)
        parity <- replicate(4, 
        {
            x <- G[sample.int(nrow(G), round(npr)), ]
            x <- t(colSums(x) %% 2)
            x[x<1] <- -1
            x
        }, FALSE)
        gbs[paste0('p', LETTERS[1:3])] <- parity[1:3]
        gbs['p'] <- parity[4]
    }
    gbs <- lapply(gbs, t)

    ## prepare model matrix
    mm <- model.matrix(tms, gbs)
    rownames(mm) <- colnames(G)

    ## fix term names
    cn <- gsub('[I()]|\\[|\\]', '', colnames(mm))
    cn <- sapply(strsplit(cn, ':'), function(x) paste0(sort(x), collapse=':'))
    cn <- sub('([a-z]+)(\\^[0-9])([0-9]+)', '\\1\\3\\2', cn)
    
    ## remove duplicated terms
    assign <- attr(mm, 'assign')
    msk <- !duplicated(cn)
    if(sum(msk) < ncol(mm))
    {
        print(sprintf('drop %d duplicated terms.', ncol(mm) - sum(msk)))
        mm <- mm[, msk, drop=F]; assign <- assign[msk]; cn <- cn[msk]
    }

    ## remove artificially created polynomial terms
    msk <- !grepl('(.*):\\1', cn)
    if(sum(msk) < ncol(mm))
    {
        print(sprintf('drop %d artificial polynomial.', ncol(mm) - sum(msk)))
        mm <- mm[, msk, drop=F]; assign <- assign[msk]; cn <- cn[msk]
    }

    ## sort the terms
    ## idx <- order(cn)
    ## mm <- mm[, idx]; assign <- assign[idx]; cn <- cn[idx]

    ## assign fixed term names
    colnames(mm) <- cn; rm(cn)

    ## keep informative columns
    msk <- apply(mm, 2, sd) > 0
    drp <- ncol(mm) - sum(msk)
    if(drp > 0 && rm.nic)
    {
        print(sprintf('drop %d non-informative columns.', drp))
        mm <- mm[, msk, drop=FALSE]
        assign <- assign[msk]
    }

    ## keep unique columns
    .nc <- ncol(mm)
    mm <- unique(mm, MARGIN=2)
    if(.nc > ncol(mm) && rm.dup)
    {
        print(sprintf('drop %d duplicated columns.', .nc-ncol(mm)))
    }

    ## crop excessive columns
    drp <- ncol(mm) - max.tms
    if(drp > 0)
    {
        ## print(sprintf('drop %d excessive columns.', drp))
        idx <- sort(sample.int(ncol(mm), max.tms))
        mm <- mm[, idx]
        assign <- assign[idx]
    }
    attr(mm, 'assign') <- assign
    mm
}

mdl.str <- function(mdl)
{
    mdl <- tail(as.character(mdl), 1)
    mdl <- gsub('[ I()ABC]', '', mdl)
    mdl <- gsub('\\+-', '-', mdl)
    mdl
}
test <- function(...)
{
    P <- 6
    g1 <- matrix(sample.int(3, P*9, T)-1, P, 9)
    rownames(g1) <- sprintf('%02d', 1:P)
    
    m1 <- gem(~ g*g[] + I(g^2), G=g1, rm.nic=T, ...)
    ## m1 <- gem(~ gA:gB:gC, G=g1, rm.nic=T)
    m1
}
