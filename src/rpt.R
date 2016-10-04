library(ggplot2)

readTraining <- function(fn)
{
    h0 <- readLines(fn, 1)
    h0 <- sub('^# ', '', h0)
    h0 <- unlist(strsplit(h0, "\t"))
    
    d0 <- read.table(fn, col.names=h0)
}

plot2T <- function(t1, t2)
{
    g <- ggplot()
    g <- g + geom_line(aes(x=log10(time), y=tcst), data=t1, color=1, linetype=1)
    g <- g + geom_line(aes(x=log10(time), y=verr), data=t1, color=1, linetype=2)
    g <- g + geom_line(aes(x=log10(time), y=tcst), data=t2, color=2, linetype=1)
    g <- g + geom_line(aes(x=log10(time), y=verr), data=t2, color=2, linetype=2)
    g
}

main <- function()
{
    t1 <- readTraining('rpt/512_032_1sg.txt')
    t2 <- readTraining('rpt/512_032_2sg.txt')

    g0 <- plot2T(t1, t2)
}
