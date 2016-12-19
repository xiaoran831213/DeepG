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

sum.all <- function(fr)
{
    dt <- cat.rpt(fr)
    su <- sum.rpt(dt)
    gp <- sum.plt(su)
    list(d=dt, s=su, g=gp)
}

main <- function(fr='sim/32')
{
    r_32 <- sum.all('raw/W08/_32HWU')
    r_64 <- sum.all('raw/W08/_64HWU')
    r128 <- sum.all('raw/W08/128HWU')
    r256 <- sum.all('raw/W08/256HWU')
    r512 <- sum.all('raw/W08/512HWU')
    r1k_ <- sum.all('raw/W08/1k_HWU')

    ggsave('rpt/r_32.png', r_32$g)
    ggsave('rpt/r_64.png', r_64$g)
    ggsave('rpt/r128.png', r128$g)
    ggsave('rpt/r256.png', r256$g)
    ggsave('rpt/r512.png', r512$g)
    ggsave('rpt/r1k_.png', r1k_$g)

    rbind(r_32$s, r_64$s, r128$s, r256$s, r512$s, r1k_$s)
}
