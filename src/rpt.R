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

plt.ssz <- function(su, grid=nhf~mdl, out=NULL)
{
    nm <- grep('^pvl', names(su), value=TRUE)
    su <- reshape(su, nm, 'pow', 'test', direction='long')
    su <- within(su, test <- sub('^pvl.', '', nm)[test])

    ## create title from constant fields
    u <- sapply(lapply(su, unique), length) == 1L
    u <- paste(names(su)[u], sapply(su[u], `[`, 1), sep='=', collapse=', ')
    
    g <- ggplot(su)
    ## g <- g + facet_wrap(~ mdl)
    if(!is.null(grid))
        g <- g + facet_grid(grid)
    
    g <- g + geom_line(aes(x=ssz, y=pow, color=test))
    g <- g + xlab('ssz')
    g <- g + ylab('pow')
    g <- g + ggtitle(u)

    vs <- all.vars(grid)
    gx <- length(unique(su$mdl))
    gy <- length(unique(su$nhf))
    gx <- 4 * gx + gx
    gy <- 4 * gy
    if(!is.null(out))
    {
        ggsave(out, g, width=gx, height=gy, limitsize=T)
    }
    g
}

plt.nhf <- function(su, grid=ssz~mdl, out=NULL)
{
    nm <- grep('^pvl', names(su), value=TRUE)
    su <- reshape(su, nm, 'pow', 'test', direction='long')
    su <- within(su, test <- sub('^pvl.', '', nm)[test])

    ## create title from constant fields
    u <- sapply(lapply(su, unique), length) == 1L
    u <- paste(names(su)[u], sapply(su[u], `[`, 1), sep='=', collapse=', ')
    
    g <- ggplot(su)
    ## g <- g + facet_wrap(~ mdl)
    if(!is.null(grid))
        g <- g + facet_grid(grid)
    
    g <- g + geom_point(aes(x=-log2(nhf), y=pow, color=test, size=1))
    g <- g + xlab('nhf')
    g <- g + ylab('pow')
    g <- g + ggtitle(u)

    vs <- all.vars(grid)
    gx <- length(unique(su$mdl))
    gy <- length(unique(su$ssz))
    gx <- 4 * gx + gx
    gy <- 4 * gy
    if(!is.null(out))
    {
        ggsave(out, g, width=gx, height=gy, limitsize=T)
    }
    g
}

sum.all <- function(fr)
{
    dt <- cat.rpt(fr)
    su <- sum.rpt(dt)
    gp <- sum.plt(su)
    list(r=dt, s=su, p=gp)
}

tmp <- function()
{
    r40 <- sum.all('sim/GMX/S40')
    write.csv(r40$s, 'rpt/s40.csv', row.names=F)
    ## r45 <- sum.all('sim/GMX/S45')
    ## r49 <- sum.all('sim/GMX/S49')
}
