#
# hold-out likelihood (Thu Aug 20 14:37:52 2015)
#

set title "hold-out likelihood"
set key bottom right
set autoscale
set grid
set xlabel "communities"
set ylabel "likelihood"
set tics scale 2
set terminal png size 1000,800
set output '.CV.likelihood.png'
plot 	".CV.likelihood.tab" using 1:2 title "" with linespoints pt 6
