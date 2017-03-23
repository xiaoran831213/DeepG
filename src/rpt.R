library(ggplot2)


cat.rpt <- function(fr)
{
    fs <- dir(fr, '.rds$', full.name=T, all.files=T)
    dt <- lapply(fs, function(x)
    {
        na.omit(readRDS(x))
    })
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
    pv.keys <- grep('^pvl', names(rp), value=T)
    dt.keys <- c(pv.keys, 'ngv', 'eot', 'eov')
    ix.keys <- names(rp)[!names(rp) %in% dt.keys]
    ix <- rp[, ix.keys]
    su <- by(rp, ix, function(r)
    {
        pow <- lapply(r[, pv.keys], .power)
        r <- data.frame(
            r[1, ix.keys],
            pow,
            itr = nrow(r),
            ## avg.eov=mean(r$eov),
            stringsAsFactors=F)
        r
    })
    su <- do.call(rbind, su)
    su
}

sum.plt <- function(su)
{
    g <- ggplot(su)
    ## g <- g + facet_wrap(~ dns)
    ## g <- g + facet_grid(ssp ~ nhf)
    g <- g + geom_line(aes(x=ssz, y=pvl.dsg), colour='black')
    g <- g + geom_line(aes(x=ssz, y=pvl.hof), colour='red')
    g <- g + facet_grid(nhv ~ str)
    g
}

sum.all <- function(fr)
{
    dt <- cat.rpt(fr)
    su <- sum.rpt(dt)
    ## gp <- sum.plt(su)
    list(d=dt, s=su)
}

tmp <- function()
{
    r0 <- sum.all('sim/W09/RRS_S00')
    r1 <- sum.all('sim/W09/RRS_S01')
    s0 <- sum.all('sim/W09/SSS_S00')
    s1 <- sum.all('sim/W09/SSS_S01')

    r <- rbind(r0$s, r1$s)
    r$str <- 'rrs'
    
    s <- rbind(s0$s, s1$s)
    s$str <- 'sss'

    rbind(r, s)
}
