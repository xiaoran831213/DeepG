library(ggplot2)


cat.rpt <- function(fr, bind=1)
{
    fs <- dir(fr, '.rds$', full.name=T, all.files=T)
    dt <- lapply(fs, function(x)
    {
        na.omit(readRDS(x))
        x <- readRDS(x)
        x
    })
    if(bind)
        dt <- do.call(rbind, dt)
    dt
}

.power <- function(p, thd=.05) sum(p < thd, na.rm=T) / sum(!is.na(p))

sum.rpt <- function(r)
{
    .power <- function(p) sum(p < .05, na.rm=T) / sum(!is.na(p))
    r$fnm <- NULL
    pv.keys <- grep('^pvl', names(r), value=T)
    dt.keys <- c(pv.keys, 'ngv', 'eot', 'eov')
    ix.keys <- names(r)[!names(r) %in% dt.keys]
    ix <- r[, ix.keys]
    su <- by(r, ix, function(r)
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
    su <- within(su,
    {
        wdp=log2(max(nhf)/nhf) + 1
    })
    su
}

plt <- function(su, x.axi='r2', grid=nhf~mdl, out=NULL)
{
    nm <- grep('^pvl', names(su), value=TRUE)
    su <- reshape(su, nm, 'pow', 'test', direction='long')
    su <- within(su, test <- sub('^pvl.', '', nm)[test])

    ## order test types
    od <- c(hof=1, rsd=2, dsg=3, ibs=4, xmx=5)
    su <- within(su, test <- reorder(test, od[test]))
    names(su)[names(su) == x.axi] = 'x'
    
    ## create title from constant fields
    .u <- sapply(lapply(su, unique), length) == 1L
    tl <- paste(names(su)[.u], sapply(su[.u], `[`, 1), sep='=', collapse=', ')

    g <- ggplot(su)
    if(!is.null(grid))
    {
        ## identify facets
        .v <- all.vars(grid)
        .t <- terms(grid)
        if(length(.v) > 1 && attr(.t, 'response') > 0)
        {
            g <- g + facet_grid(grid)
            gx <- length(unique(su[, .v[-1]]))
            gy <- length(unique(su[, .v[+1]]))
        }
        else
        {
            g <- g + facet_wrap(grid)
            gx <- length(unique(su[, .v]))
            gy <- 1
        }
    }
    else
    {
        gx <- 1
        gy <- 1
    }
    
    g <- g + geom_line(aes(x=x, y=pow, color=test))
    g <- g + xlab(x.axi)
    g <- g + ylab('pow')
    g <- g + ggtitle(tl)
    
    gx <- 3 * gx + gx
    gy <- 3 * gy + gy/2
    if(!is.null(out))
    {
        ggsave(out, g, width=gx, height=gy)
    }
    invisible(g)
}

tmp <- function()
{
    m <- mdl~wdp

    s <- sum.rpt(readRDS('rpt/rsq_gxg_gau.rds'))
    p1=plt(subset(s, T, -c(pvl.dsg, pvl.xmx)), 'r2', m, out='rpt/img/rsq_gxg_gau_ibs.png')
    p2=plt(subset(s, T, -c(pvl.ibs, pvl.xmx)), 'r2', m, out='rpt/img/rsq_gxg_gau_dsg.png')
    p3=plt(subset(s, T, -c(pvl.ibs, pvl.dsg)), 'r2', m, out='rpt/img/rsq_gxg_gau_xmx.png')

    s1 <- sum.rpt(readRDS('rpt/rsq_gxg_fp1.rds'))
    s2 <- sum.rpt(readRDS('rpt/rsq_gno_fp1.rds'))
    s <- rbind(s1, s2)
    p1=plt(subset(s, T, -c(pvl.dsg, pvl.xmx)), 'r2', m, out='rpt/img/rsq_fp1_ibs.png')
    p2=plt(subset(s, T, -c(pvl.ibs, pvl.xmx)), 'r2', m, out='rpt/img/rsq_fp1_dsg.png')
    p3=plt(subset(s, T, -c(pvl.ibs, pvl.dsg)), 'r2', m, out='rpt/img/rsq_fp1_xmx.png')

    s <- sum.rpt(readRDS('rpt/rsq_gxg_fp5.rds'))
    p1=plt(subset(s, T, -c(pvl.dsg, pvl.xmx)), 'r2', m, out='rpt/img/rsq_gxg_fp5_ibs.png')
    p2=plt(subset(s, T, -c(pvl.ibs, pvl.xmx)), 'r2', m, out='rpt/img/rsq_gxg_fp5_dsg.png')
    p3=plt(subset(s, T, -c(pvl.ibs, pvl.dsg)), 'r2', m, out='rpt/img/rsq_gxg_fp5_xmx.png')

    s <- sum.rpt(readRDS('rpt/rsq_gxg_sin.rds'))
    p1=plt(subset(s, T, -c(pvl.dsg, pvl.xmx)), 'r2', m, out='rpt/img/rsq_gxg_sin_ibs.png')
    p2=plt(subset(s, T, -c(pvl.ibs, pvl.xmx)), 'r2', m, out='rpt/img/rsq_gxg_sin_dsg.png')
    p3=plt(subset(s, T, -c(pvl.ibs, pvl.dsg)), 'r2', m, out='rpt/img/rsq_gxg_sin_xmx.png')
}
