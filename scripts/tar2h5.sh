#!/bin/bash
UNTARDIR=$SCRATCH/untar/
TARDIR=$PROJDIR/Kepler/tarfiles/
echo "Untarring Q$1 into $UNTARDIR"

case "$1" in
    1|5|7|9|10) 
	basedir="Q$1_public/"  
	;;
    2|3|4|6|8|11|12) 
	basedir="archive/data3/keplerpub/Q$1_public/"
	;;
    13|14)
	basedir="Q$1_*/"
	;;
    15)
	basedir="Q$1_EX*/"
	;;
esac

case "$1" in
    1|2|3|4|5|6|7|8|9|10|11|12)
	tar -xf $TARDIR/public_Q$1.tar -C $UNTARDIR/
	echo "here"
	;;
    13|14)
	tarfiles=$(find $TARDIR/public_Q$1_*.tar)
	n=$(echo "$tarfiles" | wc -l)
	echo "$tarfiles" | parallel -j $n tar -xf {} -C $UNTARDIR/
	basedir="Q$1_*/"
	;;
    15) 
	tarfiles=$(find $TARDIR/EX_Q15*.tgz)
	n=$(echo "$tarfiles" | wc -l)
	echo "$tarfiles" | parallel -j $n tar -xf {} -C $UNTARDIR/
	;;
esac

files=$(find $UNTARDIR/$basedir/ -name "*llc.fits")
nfiles=$(echo "$files" | wc -l)
echo "First 10 of $nfiles files"
echo "-------------------------"
(echo "$files") | head

python $KSCRIPTS/fits2h5.py $UNTARDIR/Q$1.h5 $files

echo "Removing $nfiles files"
echo "-----------------------"
rm -R $UNTARDIR/$basedir