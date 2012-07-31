"""
Calibrate a bunch of LCs.
"""
from argparse import ArgumentParser
import h5plus
import h5py
from numpy import ma
import numpy as np

import cotrend
from config import nMode

parser = ArgumentParser(description='Wrapper around detrender')

parser.add_argument('fdt',type=str,help='Light Curve File')
parser.add_argument('svd',type=str,help='SVD File')
parser.add_argument('out',nargs='?',type=str,help='output h5 file.  If none given, we replace .dt.h5 with .cal.h5')

args  = parser.parse_args()
hdt  = h5py.File(args.fdt)
hsvd  = h5py.File(args.svd)

out = h5plus.ext(args.ftd,'.cal.h5',out=args.out)

dsdt = hdt['LIGHTCURVE']
fdt  = ma.masked_array(dsdt['fdt'][:],dsdt['fmask'][:])
bv   = hsvd['V'][:nMode]

fit = ma.zeros(fdt.shape)
for i in range(fdt.shape[0]):
    p1,fit[i] = cotrend.bvfitm(fdt[i].astype(float),bv)

fcal  = fdt - fit
h5    = h5plus.File(out)

lc = np.rec.fromarrays([fcal.data,fcal.mask],names='fcal,fmask')
lc = np.array(lc)


fcal  = h5.create_dataset('LIGHTCURVE',data=lc)
kic   = h5.create_dataset('KIC' ,data=hdt['KIC'][:] )
# Simply attach the 1D collumn from the dt
ds1d = hdt['LIGHTCURVE1d']
lc1d =  h5.create_dataset(ds1d.name,data=ds1d[:])
h5.close()
