#!/usr/bin/env bash
read -ep "Enter output directory " K2_PROJDIR
BNAME=$(basename ${K2_PROJDIR})
SCRIPTDIR="${K2_PROJDIR}/scripts/"

N_STARS=$(cat ${BNAME}/pp.csv | wc -l)
N_STARS=$(expr ${N_STARS} - 1 ) # Knock off header


echo
echo "Gearing up to process ${N_STARS} light curves"
echo 

i="0"

while [ $i -lt $N_STARS ]
do
    echo ". $HOME/k2_setup.sh"
    echo "cd $K2_DIR"
    echo "export K2_PROJDIR=$K2_PROJDIR"
    echo "python ${K2_DIR}/code/py/scripts/terraWrap.py ${BNAME}/pp.csv ${BNAME}/grid.csv ${BNAME}/dv.csv $i"
    i=$[$i+1]
done > ${SCRIPTDIR}/${BNAME}.tot
