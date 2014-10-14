#!/usr/bin/env bash

echo "Setting up environment for K2"

export K2_DIR=${PROJDIR}/K2/
export K2PHOT_DIR=${HOME}/code_carver/k2phot

export PYTHONPATH=${K2_DIR}/code/py/:${PYTHONPATH}
export PYTHONPATH=${K2PHOT_DIR}/code/py/:${PYTHONPATH}
export PATH=${PATH}:${K2_DIR}/code/bin/
export PATH=${PATH}:${K2_DIR}/code/py/
export K2PHOTFILES=${PROJDIR}/www/k2photfiles/
cd $K2_DIR

