#!/usr/bin/env bash
STARNAME=$1

. ${HOME}/k2_setup.sh
cd ${K2_DIR}


PHOTDIR=${PROJDIR}/www/K2/photometry/C0_12-14
TPSDIR=${PROJDIR}/www/K2/TPS/C0_12-14
OUTDIR=${TPSDIR}/output/${STARNAME}

LCFILE=${PHOTDIR}/output/${STARNAME}/${STARNAME}.h5
GRIDFILE=${OUTDIR}/${STARNAME}.grid.h5

mkdir -p ${OUTDIR}

python ${K2_DIR}/code/py/scripts/terraWrap.py pp ${LCFILE} ${GRIDFILE} pars.sqlite ${STARNAME}
python ${K2_DIR}/code/py/scripts/terraWrap.py grid ${GRIDFILE} pars.sqlite ${STARNAME}

python ${K2_DIR}/code/py/scripts/terraWrap.py dv ${GRIDFILE} pars.sqlite ${STARNAME}

chmod -R o+rX ${OUTDIR}

python ${K2_DIR}/code/py/scrape_terra.py ${GRIDFILE} ${TPSDIR}/scrape.db
