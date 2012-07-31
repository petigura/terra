"""
Calibrate a bunch of LCs.
"""
from argparse import ArgumentParser
import h5plus
import h5py
from numpy import ma
import numpy as np
from matplotlib import mlab

import cotrend
from config import nMode

parser = ArgumentParser(description='Wrapper around detrender')

parser.add_argument('fdt',type=str,help='Light Curve File')
parser.add_argument('svd',type=str,help='SVD File')
parser.add_argument('out',nargs='?',type=str,help='output h5 file.  If none given, we replace .dt.h5 with .cal.h5')

args  = parser.parse_args()
hdt  = h5py.File(args.fdt)
hsvd  = h5py.File(args.svd)

out = h5plus.ext(args.fdt,'.cal.h5',out=args.out)

rdt = hdt['LIGHTCURVE'][:]
fdt  = ma.masked_array(rdt['fdt'][:],rdt['fmask'][:])
bv   = hsvd['V'][:nMode]

fit    = ma.zeros(fdt.shape)
p1,fit = cotrend.bvfitm(fdt.astype(float),bv)
fcal  = fdt - fit
rcal = mlab.rec_append_fields(rdt,['fit','fcal'],[fit,fcal])

h5    = h5plus.File(out)
fcal  = h5.create_dataset('LIGHTCURVE',data=rcal)
h5.close()
