#!/usr/bin/env bash
read -ep "Enter output directory " K2_PROJDIR
BNAME=$(basename ${K2_PROJDIR})

GRIDDIR="${K2_PROJDIR}/grid/"
SCRIPTDIR="${K2_PROJDIR}/scripts/"
echo "grid structures will be stored here: ${GRIDDIR}"
echo "plots will be stored here: ${GRIDDIR}"
echo "scripts will be stored here: ${SCRIPTDIR}"

if [ -d "$K2_PROJDIR" ] 
then
    echo "Directory exists, could be overwriting files" 
else
    echo "creating directories"
    mkdir -p ${GRIDDIR} ${SCRIPTDIR}
fi

cp tpstemp.ipynb ${K2_PROJDIR}/

echo
echo "cd ${K2_PROJDIR}"
echo "ipython notebook tpstemp.ipynb"
echo 
echo "create pp.csv, grid.csv, and dv.csv"

