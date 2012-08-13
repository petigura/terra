"""
Wrapper around the spline detrending.
"""
from argparse import ArgumentParser
import h5plus
import h5py
import qalg
import numpy as np
from numpy import ma
import cotrend
import sys
from config import nMode,nModeSave
import os

parser = ArgumentParser(description='Perform Robust SVD')
parser.add_argument('out', type=str,help='file storing Pricip Comp.')
parser.add_argument('inp', nargs='+',type=str ,help='input dt files')

args  = parser.parse_args()
out   = args.out
inp   = args.inp

print "Loading %i ensemble files" % len(inp)
fdtL = []
kicL = []
for f in inp:
    try:
        h = h5py.File(f,'r+')
        lc = h['LIGHTCURVE'][:]
        fdt = ma.masked_array(lc['fdt'],lc['fmask'])
        fdtL.append( fdt )
        kicL.append( os.path.basename(f).split('.')[0] )
        h.close()
    except:
        print sys.exc_info()[1]
        pass

fdt = ma.vstack(fdtL)
kic = np.hstack(kicL)
print "Sucessfully loaded %i light curves" % fdt.shape[0]

U,S,V,A,fit,kic = cotrend.mkModes(fdt,kic)

h5 = h5plus.File(out)
h5.create_dataset('U'     ,data=U,compression='lzf')
h5.create_dataset('S'     ,data=S,compression='lzf',shuffle=True)
h5.create_dataset('V'     ,data=V[:nModeSave],compression='lzf',shuffle=True)
h5.create_dataset('A'     ,data=A[:,:nModeSave] )
h5.create_dataset('fit'   ,data=fit)
h5.create_dataset('KIC'   ,data=kic)
h5.close()
print "mkModes: created %s" % out
