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
    su <- aggregate(pvl ~ ssz + use + knl, data=rp, FUN=.power)
    su
}

sum.plt <- function(su)
{
    g <- ggplot(su, aes(x=ssz, y=pvl))
    g <- g + geom_line(aes(colour=knl, linetype=use))
    g
}

main <- function(fr='sim/32')
{
    dt <- cat.rpt(fr)
    su <- sum.rpt(dt)
    gp <- sum.plt(su)
    list(d=dt, s=su, g=gp)
}
