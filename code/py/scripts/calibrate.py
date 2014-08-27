"""
Take raw photometry from path_phot, and applies the calibrations from 

"""
import matplotlib
matplotlib.use('Agg')

from argparse import ArgumentParser
import pandas as pd
import numpy as np
from numpy import ma
from matplotlib.pylab import *

import h5plus
import h5py

import cotrend
import sys
from config import path_phot,path_train
import stellar
import photometry
import prepro


parser = ArgumentParser(description='Ensemble calibration')

args  = parser.parse_args()

# Load up the information from ensemble cotrending
path_train_dfA = path_train.replace('train','train-dfA')
path_cal = path_train.replace('train','cal')

with h5plus.File(path_train) as h5:
    for k in 'U V U_clip'.split():
        exec "%s = h5['%s'][:]" % (k,k)

dfA = pd.read_hdf(path_train_dfA,'dfA')
dfAc_st = pd.read_hdf(path_train_dfA,'dfAc_st')

# Load up stars
dfstars = photometry.phot_vs_kepmag()
nstars = len(dfstars)
print "calibrating %s stars" % nstars
def cal(i):
    ind = dict(dfstars.iloc[i])
    name = ind['epic']
    lc = photometry.read_phot(path_phot, name)
    lc = prepro.rdt(lc) # detrend light curve
    fdt = ma.masked_array(lc['fdt'],lc['fmask'])

    A,fit = cotrend.bvfitm(fdt,U.T)
    fit = fit.data

    dfAc_st['Afit'] = A
    inlier = np.all(dfAc_st.Afit.between(dfAc_st.outlo,dfAc_st.outhi))
    if inlier:
        fcal = fdt - fit
    else:
        fcal = fdt.copy()
        fit = np.zeros(fcal.size)

    # Add in fit and fcal fields
    lc = pd.DataFrame(lc)
    lc['fit'] = fit
    lc['fcal'] = fcal.data

    lc['cr'] = prepro.isCR(lc['fcal'])
    lc['fmask'] = lc['fmask'] | lc['cr']

    lc = np.array(lc.to_records(index=False))
    
    outd = dict(ind,**dict(dfAc_st.Afit))
    outd['inlier'] = inlier
    return lc,outd

lc0,outd = cal(0)

with h5plus.File(path_cal) as h5:
    h5.create_dataset('cal', dtype=lc0.dtype, shape=(nstars, lc0.size))
    h5['name'] = np.array(dfstars.epic)
    outdL = []
    
    for i in range(nstars):
        lc,outd = cal(i)
        h5['cal'][i,:] = lc
        outdL+=[outd]
        if (i%100)==0:
            print i

    
dfstars = pd.DataFrame(outdL)
path_cal_meta = path_cal.replace('cal','cal-meta')
dfstars.to_hdf(path_cal_meta,'dfstars')
