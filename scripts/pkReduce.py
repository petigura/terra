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

prsr = ArgumentParser()
prsr.add_argument('--inp',nargs='+',type=str, help='input .pk.h5 files.  Must contain KIC attr')
prsr.add_argument('--out',type=str, help='output database')
args  = prsr.parse_args()

print "reducing %i files" % len(args.inp)
print "out: %s" % args.out
rL = []
for f in args.inp:
    h5 = h5py.File(f)
    res = h5['RES'][:]
    res = mlab.rec_append_fields(res,'KIC',h5.attrs['KIC'])
    res = mlab.rec_append_fields(res,'P',res['Pcad']*keptoy.lc)
    rL.append(res)
    h5.close()
rL = np.hstack(rL)
tpk = qalg.rec2tab(rL)
tpk.table_name = 'pk'
tpk.write('sqlite',args.out,overwrite=True)
