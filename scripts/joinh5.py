"""
Calibrate a bunch of LCs.
"""
from argparse import ArgumentParser
import h5plus
import h5py
from numpy import ma
import numpy as np
import cotrend
import keplerio
from matplotlib import mlab

desc="""
Join Dataset By Stacking Rows

`LIGHTCURVE` is vstacked
`KIC` is hstacked
"""
prsr = ArgumentParser(description=desc)
prsr.add_argument('--inp',nargs='+' ,type=str , help='input files')
prsr.add_argument('--out',type=str,help='output h5 file.')
args  = prsr.parse_args()

files = args.inp
h0    = h5py.File(files[0])

# Figure out which datasets we vtacking
kTwoDim = ''
for k in h0.iterkeys():
    if len(h0[k].shape) is 2:
        kTwoDim = k

TwoDimdtype = h0[kTwoDim].dtype
kicdtype    = h0['KIC'].dtype
ncols = h0[kTwoDim].shape[1]
h0.close()

nrows = 0 
for f in files:
    h = h5py.File(f)
    nrows += h[kTwoDim].shape[0]
    h.close()

h5out = h5plus.File(args.out)
dsout = h5out.create_dataset(kTwoDim,shape=(nrows,ncols,),dtype=TwoDimdtype,
                             compression='lzf',shuffle=True)
dskic = h5out.create_dataset('KIC',shape=(nrows,),dtype=kicdtype)

irow = 0 
for f in files:
    h = h5py.File(f)
    dsSub = h[kTwoDim]
    nSubRow = dsSub.shape[0]
    dsout[irow:irow+nSubRow] = dsSub[:]
    dskic[irow:irow+nSubRow] = h['KIC'][:]
    h.close()

h5out.close()
