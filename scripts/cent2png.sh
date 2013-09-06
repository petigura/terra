#!/usr/bin/env bash

#
# Converts the pdf files into png files for quick viewing
#

DV_DIR=$PWD
echo $DV_DIR
tpng1=$DV_DIR/$1.png1
tpng2=$DV_DIR/$1.png2
opng=$DV_DIR/${1%.pdf}.png

convert -density 600 $1 -resize 25% -trim $tpng1
convert $tpng1 -crop 600x300+1000+1200 -resize 200% $tpng2
convert +append $tpng1 $tpng2 $opng

