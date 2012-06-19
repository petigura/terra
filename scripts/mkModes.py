"""
Wrapper around the spline detrending.
"""

from argparse import ArgumentParser
import h5plus
import h5py
import qalg
import keplerio
import prepro
import numpy as np
from numpy import ma
import detrend
from matplotlib import mlab
import cotrend

nModesSave = 10
nMode=4
sigOut=10
maxit=10

parser = ArgumentParser(description='Perform Robust SVD')
parser.add_argument('inp',  type=str)
parser.add_argument('out',  type=str,help='output h5 file')

args  = parser.parse_args()
out   = args.out
inp   = args.inp

h5inp = h5py.File(inp)
ds    = h5inp['LIGHTCURVE']
fdt   = ds['fdt'][:]
kic   = h5inp['KIC'][:]

# Cut out the bad columns and rows
fdt[np.isnan(fdt)] = 0
czero = fdt.sum(axis=0)!=0.
fdt = fdt[:,czero] 

rzero = np.median(fdt,axis=1)!=0.
fdt = fdt[rzero,:] 
kic = kic[rzero]


# Normalize by Median Absolute Dev.  Normalized reduced Chi2 should be about 1
mad = ma.median(ma.abs(fdt),axis=1)
mad = mad.reshape(mad.size,1)
fdt = fdt/mad
fdt = np.vstack(fdt)

U,S,Vtemp,goodid,X2 = \
    cotrend.robustSVD(fdt,nMode=nMode,sigOut=sigOut,maxit=maxit)

V = np.zeros((Vtemp.shape[0],ds.shape[1]))

V[:,czero] = Vtemp

h5 = h5plus.File(out)
h5.create_dataset('U'     ,data=U)
h5.create_dataset('S'     ,data=S)
h5.create_dataset('V'     ,data=V,compression='lzf',shuffle=True)
h5.create_dataset('MAD'   ,data=mad)
h5.create_dataset('goodid',data=goodid)
h5.create_dataset('KIC'   ,data=kic)
h5.create_dataset('X2'    ,data=X2)

# For space reasons, think about only saving the priciple components that I care about

h5.close()
