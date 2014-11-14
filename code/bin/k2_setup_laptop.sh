#!/usr/bin/env bash

echo "Setting up environment for K2"

export K2_DIR=${HOME}/Marcy/K2
export K2PHOT_DIR=${HOME}/Marcy/k2phot

export PYTHONPATH=${K2_DIR}/code/py/:${PYTHONPATH}
export PYTHONPATH=${K2PHOT_DIR}/code/py/:${PYTHONPATH}
export PATH=${PATH}:${K2_DIR}/code/bin/
export PATH=${PATH}:${K2_DIR}/code/py/
export K2PHOTFILES=${K2PHOT_DIR}/k2photfiles/
export K2WEBAPP_DB=${K2_DIR}/scrape.db

cd $K2_DIR

