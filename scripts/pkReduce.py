#!/usr/bin/env python
"""
Take a list of pk files and merge.
"""
from argparse import ArgumentParser
import h5py
import atpy
import qalg
from matplotlib import mlab
import numpy as np
import keptoy
import os
from tval import readPkScalar

prsr = ArgumentParser()
prsr.add_argument('out',type=str, help='output database')
prsr.add_argument('inp',nargs='+',type=str, help='input .pk.h5 files.  Must contain KIC attr')
args  = prsr.parse_args()
files = args.inp
print "reducing %i files" % len(files)

res = readPkScalar(files[0])
rL  = []
for i in range(len(args.inp)):
    try:
        rL.append( readPkScalar(files[i]) )
    except:
        print "%s failed" % files[i]
        pass

rL = np.hstack(rL)

tpk = qalg.rec2tab(rL)
tpk.table_name = 'pk'
tpk.write('sqlite',args.out,overwrite=True)
print "pkReduce: %s" % args.out
