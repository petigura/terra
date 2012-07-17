#!/usr/bin/env python

"""
Split h5 file into individual files
"""
from argparse import ArgumentParser
import h5plus
import h5py
from numpy import ma
import numpy as np
import cotrend
import keplerio
from matplotlib import mlab

prsr = ArgumentParser()
prsr.add_argument('inp',type=str, help='input files')
args  = prsr.parse_args()

file = args.inp
h0    = h5py.File(file)

# Figure out which datasets we vtacking
kTwoDim = ''
for k in h0.iterkeys():
    if len(h0[k].shape) is 2:
        kTwoDim = k

TwoDimdtype = h0[kTwoDim].dtype
kicdtype    = h0['KIC'].dtype
nrows       = h0['KIC'].shape[0]
for i in range(nrows):
    ext = file[file.find('.'):]
    kic = h0['KIC'][i]
    fout = "%09d%s" % (kic,ext)
    print fout

    h = h5plus.File(fout)
    h.create_dataset(kTwoDim,data=h0[kTwoDim][i])
    h.close()

h0.close()
