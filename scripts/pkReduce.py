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
import pandas
prsr = ArgumentParser()
prsr.add_argument('out',type=str, help='output database')
prsr.add_argument('inp',nargs='+',type=str, help='input .pk.h5 files.  Must contain KIC attr')
args  = prsr.parse_args()
files = args.inp
print "reducing %i files" % len(files)

dL  = []
for i in range(len(args.inp)):
    if np.mod(i,100) == 0:
        print "%i files reduced" % i 
    f = files[i]
    print f
    try:
        p = tval.Peak(f,quick=True)
        d = pandas.DataFrame(p.get_db())
        d['bname'] = os.path.basename(f).split('.')[0]
        dL.append( d )
    except:
        print "%s failed" % f
        pass
dL = pandas.concat(dL)
f = open(args.out,'w')
f.writelines( dL.to_string() )
f.close()
print "pkReduce: %s" % args.out
