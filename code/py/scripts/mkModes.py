"""
Wrapper around the spline detrending.
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
from config import path_phot
import stellar
import photometry
import prepro

algo = 'ICA'
n_components = 8
plot_basename = 'cotrend_robust%s_ncomp=%i' % (algo,n_components) 

parser = ArgumentParser(description='Ensemble calibration')
parser.add_argument('path_phot',type=str ,help='Path to photometry database')
parser.add_argument('path_train',type=str ,help='Path to photometry database')
args  = parser.parse_args()

# Grab stars used for computing modes
df0 = photometry.phot_vs_kepmag()
dftr = df0.copy()
dftr = dftr[dftr.logfmed_resid.between(-0.5,0.5)]
#dftr = dftr[dftr.prog.str.contains('GKM|cool')]
dftr = dftr.query('10 < kepmag < 13')

# Load up light curves
lc = photometry.read_phot(path_phot, list(dftr.epic) )
lc = np.vstack([prepro.rdt(lci) for lci in lc]) # Detrend Light curves
fdt = ma.masked_array(lc['fdt'],lc['fmask'])

# Mask out bad columns and rows
intmask = (~fdt.mask).astype(int) # Mask as integers
bcol = intmask.sum(axis=0)!=0 # True - good column
brow = intmask.sum(axis=1)!=0 # True - good row
fdt = fdt[:,bcol]

for n,b in zip(['cadences','light curves'],[bcol,brow]):
    print "%i %s masked out" % (b.size - b.sum(),n)
print ""
    
fdt_sig = 1.48 * median(abs(fdt),axis=1)
fdt_stand = fdt / fdt_sig[:,newaxis]
M = fdt_stand.data.T.copy()

U,V,icolin = cotrend.robust_components(M,algo=algo,n_components=n_components)
cotrend.plot_PCs(U,V)

U_clip = U.copy()
U = np.zeros((bcol.size,U.shape[1]))
U[bcol,:] = U_clip

for kepmag in [10,11,12,14]:
    clf()
    cotrend.plot_modes_diag(U,V,kepmag)
    gcf().savefig( plot_basename + '_kepmag=%i.png' % kepmag)

M_clip = M[:,icolin]
fdt_clip = fdt[icolin] # Clipout light curves that didn't make the
nclip = len(icolin)
A = [cotrend.bvfitm(fdt_clip[i],U_clip.T)[0] for i in range(nclip)]
A = ma.vstack(A)

fdt_stand_clip = fdt_stand[icolin]
A_stand = [cotrend.bvfitm(fdt_stand_clip[i],U_clip.T)[0] for i in range(nclip)]
A_stand = ma.vstack(A_stand)

dftr_clip = dftr.iloc[icolin]

kAs = ['A%02d' % i for i in range(n_components)]
dfA = pd.DataFrame(A,columns=kAs,index=dftr_clip.index)
dfA = pd.concat([dftr_clip,dfA],axis=1)

# Label points as inliers
dfAc = cotrend.dfA_get_coeffs(dfA)
dfAc_st = pd.DataFrame(dfAc.median(),columns=['med'])
dfAc_st['sig'] = 1.48*abs(dfAc - dfAc_st['med']).median()
dfAc_st['outhi'] = dfAc_st['med'] + 5 * dfAc_st['sig']
dfAc_st['outlo'] = dfAc_st['med'] - 5 * dfAc_st['sig']
inlier = (dfAc_st.outlo < dfAc) & (dfAc < dfAc_st.outhi)
dfA['inlier'] = np.all(inlier,axis=1)

print ""
print "%i stars in training set " % fdt.shape[0]
print "%i stars survived iterative %s " % (fdt_clip.shape[0],algo)
print "%i stars have inlier coefficients" % dfA.inlier.sum()

clf()
pd.scatter_matrix(dfA[dfA.inlier][kAs],figsize=(8,8))
gcf().savefig(plot_basename+'_coeffs.png')

clf()
cotrend.plot_mode_FOV(dfA[dfA.inlier])
gcf().set_tight_layout(True)
gcf().savefig(plot_basename+'_FOV.png')

path_train = args.path_train
path_train_dfA = path_train.replace('train','train-dfA')
dfA.to_hdf(path_train_dfA,'dfA')
dfAc_st.to_hdf(path_train_dfA,'dfAc_st')

with h5plus.File(path_train) as h5:
    for k in 'U V U_clip'.split():
        exec "h5.create_dataset('%s',data=%s)" % (k,k)
