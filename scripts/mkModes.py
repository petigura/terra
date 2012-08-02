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

from config import nMode,nModeSave

parser = ArgumentParser(description='Perform Robust SVD')
parser.add_argument('inp', type=str ,help='input h5 file')
parser.add_argument('out', nargs='?',type=str,help='output h5 file.  If not given, we just change the extention from dt.h5 to svd.h5')

args  = parser.parse_args()
out   = args.out
inp   = args.inp
if out is None:
    out = inp.replace('dt.h5','svd.h5')

h5inp = h5py.File(inp)
ds    = h5inp['LIGHTCURVE']
fdt   = ma.masked_array(ds['fdt'][:],ds['fmask'][:],fill_value=np.nan)
kic   = h5inp['KIC'][:]

U,S,V,A,fit,kic = cotrend.mkModes(fdt,kic)

h5 = h5plus.File(out)
h5.create_dataset('U'     ,data=U,compression='lzf')
h5.create_dataset('S'     ,data=S,compression='lzf',shuffle=True)
h5.create_dataset('V'     ,data=V[:nModeSave],compression='lzf',shuffle=True)
h5.create_dataset('A'     ,data=A)
h5.create_dataset('fit'   ,data=fit)
h5.create_dataset('KIC'   ,data=kic)

h5.close()
