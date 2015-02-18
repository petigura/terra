#!/usr/bin/env python

"""
Split h5 file into individual files
"""
from argparse import ArgumentParser
import h5plus
import h5py
import os

from matplotlib import mlab

prsr = ArgumentParser()
prsr.add_argument('inp',type=str, help='input files')
args  = prsr.parse_args()

finp  = args.inp
h0    = h5py.File(finp)
dir   = os.path.dirname(finp)

# Figure out which datasets we vtacking
kTwoDim = ''
for k in h0.iterkeys():
    if len(h0[k].shape) is 2:
        kTwoDim = k

TwoDimdtype = h0[kTwoDim].dtype
kicdtype    = h0['KIC'].dtype
nrows       = h0['KIC'].shape[0]

for i in range(nrows):
    ext = finp[finp.find('.'):]
    kic = h0['KIC'][i]
    fout = "%09d%s" % (kic,ext)
    pout = os.path.join(dir,fout)
    print fout

    h = h5plus.File(pout)
    h.create_dataset(kTwoDim,data=h0[kTwoDim][i])
    h.close()

h0.close()
