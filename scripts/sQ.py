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

parser = ArgumentParser(description='Stictch Calibrated Light Curves Together')
parser.add_argument('files',nargs='+',type=str , help='input files')
parser.add_argument('--out',type=str,help='output h5 file.  If it is not set, ')

args  = parser.parse_args()
h5L = [h5py.File(f) for f in args.files]
h5out = h5plus.File(args.out)

# Find the kic stars that appear in all quarters
# Initial values of the joined fcal array and the kic array.

kicj  = h5L[0]['KIC'][:]
nh5 = len(h5L)
for i in range(nh5-1):
    h5 = h5L[i]
    dslc = h5['LIGHTCURVE']
    print "%s has %i rows" % (h5.filename,dslc.shape[0])
    k2 = h5L[i+1]['KIC'][:]
    x1 = np.arange(kicj.size)
    x2 = np.arange(k2.size)
    x1j,x2j,kicj = cotrend.join_on_kic(x1,x2,kicj,k2)

# Output for the last column
h5 = h5L[i+1]
dslc = h5['LIGHTCURVE']
print "%s has %i rows" % (h5.filename,dslc.shape[0])
print "%i stars appear in all quarters" % kicj.size

# Select the rows from fcal that correspond to the kicj stars
lcL = []
for i in range(nh5):
    h5 = h5L[i]
    k1   = h5['KIC'][:]
    lc   = h5['LIGHTCURVE'][:]
    x2   = np.arange(kicj.size)
    lc,x2j,kicj = cotrend.join_on_kic(lc,x2,k1,kicj)
    lcL.append(lc)

# For each row, stich the quarters together
rLCL = []
for i in range(lcL[0].shape[0]):
    rL = []
    for j in range(nh5):
        ds1d = h5L[j]['LIGHTCURVE1d']
        lc = lcL[j][i]
        names = ['t','cad']
        arrs  = [ds1d[n] for n in names]
        lc = mlab.rec_append_fields(lc,names,arrs)

        rL.append(lc)
    rLC = keplerio.rsQ(rL)
    rLCL.append(rLC)

rLCL = np.vstack(rLCL)

h5out.create_dataset('LIGHTCURVE',data=rLCL,compression='lzf',shuffle=True)
h5out.create_dataset('LIGHTCURVE1d',data=ds1d[:])
h5out.create_dataset('KIC' ,data=kicj )
h5out.close()
