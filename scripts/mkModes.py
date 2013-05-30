"""
Wrapper around the spline detrending.
"""
from argparse import ArgumentParser
import h5plus
import h5py
import numpy as np
from numpy import ma
import cotrend
import sys
from config import nMode,nModeSave

parser = ArgumentParser(description='Perform Robust SVD')
parser.add_argument('dtfile', type=str ,help='dt')

args  = parser.parse_args()

h5  = h5py.File(args.dtfile)
dt  = h5['dt']
fdt = ma.masked_array( dt['fdt'], dt['fmask'] )
kic = h5['kic'][:]

bgood = kic > 0
print "%i  / %i good" % (kic[bgood].size, kic.size )
kic = kic[bgood]
fdt = fdt[bgood,:]

U,S,V,A,fit,kic = cotrend.mkModes(fdt,kic)
out = args.dtfile.replace('dt','svd')
h5 = h5plus.File(out)
h5.create_dataset('U'     ,data=U,compression='lzf')
h5.create_dataset('S'     ,data=S,compression='lzf',shuffle=True)
h5.create_dataset('V'     ,data=V[:nModeSave],compression='lzf',shuffle=True)
h5.create_dataset('A'     ,data=A[:,:nModeSave] )
h5.create_dataset('KIC'   ,data=kic)
h5.close()
print "mkModes: created %s" % out
