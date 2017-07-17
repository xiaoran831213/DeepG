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
    su <- subset(su, itr > 100)
    su
}

plt.ssz <- function(su, grid=-log2(nhf)~mdl, out=NULL)
{
    nm <- grep('^pvl', names(su), value=TRUE)
    su <- reshape(su, nm, 'pow', 'test', direction='long')
    su <- within(su, test <- sub('^pvl.', '', nm)[test])

    ## order test types
    od <- c(hof=1, rsd=2, dsg=3, ibs=4, xmx=5)
    su <- within(su, test <- reorder(test, od[test]))
    
    ## create title from constant fields
    .u <- sapply(lapply(su, unique), length) == 1L
    .u <- paste(names(su)[.u], sapply(su[.u], `[`, 1), sep='=', collapse=', ')

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
    
    g <- g + geom_line(aes(x=ssz, y=pow, color=test))
    g <- g + xlab('ssz')
    g <- g + ylab('pow')
    g <- g + ggtitle(.u)
    
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
    r <- readRDS('rpt/s4_.rds')
    s <- sum.rpt(r)
    m=mdl~log2(2^10/nhf)
    p=plt.ssz(subset(s, mdl=='g',   -c(pvl.dsg, pvl.xmx)), m, out='rpt/img/szz_gno_ibs.png')
    p=plt.ssz(subset(s, mdl=='g*g', -c(pvl.dsg, pvl.xmx)), m, out='rpt/img/szz_gxg_ibs.png')
    p=plt.ssz(subset(s, mdl=='g',   -c(pvl.ibs, pvl.xmx)), m, out='rpt/img/szz_gno_dsg.png')
    p=plt.ssz(subset(s, mdl=='g*g', -c(pvl.ibs, pvl.xmx)), m, out='rpt/img/szz_gxg_dsg.png')
}
