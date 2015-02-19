#!/usr/bin/env bash

echo "Setting up environment for K2"

export K2_DIR=${PROJDIR}/K2/
export K2PHOT_DIR=${HOME}/code_carver/k2phot

export PYTHONPATH=${K2_DIR}/code/py/:${PYTHONPATH}
export PYTHONPATH=${K2PHOT_DIR}/code/py/:${PYTHONPATH}
export PATH=${PATH}:${K2_DIR}/code/bin/
export PATH=${PATH}:${K2_DIR}/code/py
export PATH=${PATH}:${K2PHOT_DIR}/code/bin/
export PATH=${K2_DIR}/k2_webapp/:${PATH}
export K2PHOTFILES=${PROJDIR}/www/k2photfiles/
export K2WEBAPP_DB=${PROJDIR}/www/K2/TPS/C0_10-10/scrape.db
export K2WEBAPP_HOST="0.0.0.0"
export K2_ARCHIVE=${PROJDIR}/www/K2/

cd $K2_DIR

