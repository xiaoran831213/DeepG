library(ggplot2)


cat.rpt <- function(fr)
{
    fs <- dir(fr, '.rds$', full.name=T, all.files=T)
    dt <- lapply(fs, readRDS)
    dt <- do.call(rbind, dt)
    dt
}

sum.rpt <- function(rp)
{
    .power <- function(p)
    {
        sum(p < .05, na.rm=T) / sum(!is.na(p))
    }
    rp$seq <- NULL
    dt.keys <- c('pvl.gno', 'pvl.hof', 'ngv', 'eot', 'eov')
    ix.keys <- names(rp)[!names(rp) %in% dt.keys]
    ix <- rp[, ix.keys]
    su <- by(rp, ix, function(r)
    {
        r <- data.frame(
            r[1, ix.keys],
            pow.gno=.power(r$pvl.gno),
            pow.hof=.power(r$pvl.hof),
            avg.ngv=mean(r$ngv),
            avg.eov=mean(r$eov),
            stringsAsFactors=F)
        r
    })
    su <- do.call(rbind, su)
    su
}

sum.plt <- function(su)
{
    g <- ggplot(su)
    ## g <- g + geom_line(aes(x=ssz, y=pow.hof, color=as.factor(ssp)))
    ## g <- g + facet_wrap(~ wdp)
    ## g <- g + facet_grid(ssp ~ wdp)
    g <- g + geom_line(aes(x=ssz, y=pow.hof, colour=as.factor(wdp)))
    g <- g + facet_wrap(~ ssp)
    g
}

sum.all <- function(fr)
{
    dt <- cat.rpt(fr)
    su <- sum.rpt(dt)
    gp <- sum.plt(su)
    list(d=dt, s=su, g=gp)
}

main <- function()
{
    rNE <- sum.all('sim/W08/40_ET1')
    rCV <- sum.all('sim/W08/40_CV4')

    ggsave('rpt/r_ne.png', rNE$g)
    ggsave('rpt/r_cv.png', rCV$g)
    rbind(rNE$s, rCV$s)
}
