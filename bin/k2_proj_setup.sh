#!/bin/bash
#export K2_PROJ=$1
#
#case $K2_PROJ in
#    Ceng2C0)
#	K2_CAMP=Ceng
#	;;
#esac
#
#export K2_PROJDIR=$K2_DIR/$K2_PROJ
#export K2_CAMP=$K2_CAMP
#
#echo "K2_PROJ=$K2_PROJ"
#echo "K2_PROJDIR=$K2_PROJDIR"
#echo "K2_CAMP=$K2_CAMP"

# Directory containing fits files
export K2_CAMP=Ceng
export K2_PIX_DIR=$K2_DIR/pixel/$K2_CAMP/
export K2_PHOT_DIR=$K2_DIR/photometry/Ceng2C0/
export K2_SEARCH_DIR=$K2_DIR/search/TPS-Ceng2C0/

