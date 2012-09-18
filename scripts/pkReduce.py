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
import tval

prsr = ArgumentParser()
prsr.add_argument('out',type=str, help='output database')
prsr.add_argument('inp',nargs='+',type=str, help='input .pk.h5 files.  Must contain KIC attr')
args  = prsr.parse_args()
files = args.inp
print "reducing %i files" % len(files)

rL  = []
for i in range(len(args.inp)):
    if np.mod(i,100) == 0:
        print "%i files reduced" % i 
    f = files[i]
    try:
        p = tval.Peak(f,quick=True)
        rL.append( p.get_db() )
    except:
        print "%s failed" % f
        pass
rL = np.hstack(rL)

tpk = qalg.rec2tab(rL)
tpk.table_name = 'pk'
tpk.write('sqlite',args.out,overwrite=True)
print "pkReduce: %s" % args.out
