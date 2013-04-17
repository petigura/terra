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
parser.add_argument('q', type=int,help='file storing Pricip Comp.')
parser.add_argument('inp', nargs='+',type=str ,help='input dt files')

args  = parser.parse_args()
out   = args.out
inp   = args.inp
q = args.q

print "Loading %i ensemble files" % len(inp)
fdtL = []
kicL = []


i = 0
for f in inp:
    try:
        with h5py.File(f) as h5:
            qstr = 'Q%i' % q
            fdt  = h5['dt'][qstr]['fdt'][:]
            mask = h5['raw'][qstr]['fmask'][:]
            fdt = ma.masked_array(fdt,mask)
            fdtL.append(fdt)
            kic = int(f.split('/')[-1].split('.')[0])
            kicL.append(kic)
    except:
        print sys.exc_info()[1]
    i+=1
    if i % 100 ==0:
        print i

fdt = ma.vstack(fdtL)
kic = np.hstack(kicL)
print "Sucessfully loaded %i light curves" % fdt.shape[0]
U,S,V,A,fit,kic = cotrend.mkModes(fdt,kic)

h5 = h5plus.File(out)
h5.create_dataset('U'     ,data=U,compression='lzf')
h5.create_dataset('S'     ,data=S,compression='lzf',shuffle=True)
h5.create_dataset('V'     ,data=V[:nModeSave],compression='lzf',shuffle=True)
h5.create_dataset('A'     ,data=A[:,:nModeSave] )
h5.create_dataset('KIC'   ,data=kic)
h5.close()
print "mkModes: created %s" % out
