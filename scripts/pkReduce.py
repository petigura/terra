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

prsr = ArgumentParser()
prsr.add_argument('out',type=str, help='output database')
prsr.add_argument('inp',nargs='+',type=str, help='input .pk.h5 files.  Must contain KIC attr')
args  = prsr.parse_args()

files = args.inp
print "reducing %i files" % len(files)
print "out: %s" % args.out



akeys =['tdur','p99','df','s2n','t0','P','p90','p50']
names = ['sKIC','pp','tau','b','pp180','tau180','b180','maQSES','madSES']+akeys
         

dtype = zip(names,['|S10']+[float]*(len(names)-1))


def read(f):
    import pdb;pdb.set_trace()
    h5 = h5py.File(f)
    g = h5['/pk0']
    print f

    res = np.zeros(1,dtype=dtype)
    # Read the fields stored in pk file.
    adict = dict(g.attrs)
    for k in akeys:
        res[k] = adict[k]    
    res['sKIC'] = os.path.basename(f).split('.')[0]
    res['pp']      = g['pL1'][0]
    res['tau']     = g['pL1'][1]
    res['b']       = g['pL1'][2]
    res['pp180']   = g['pL1_180'][0]
    res['tau180']  = g['pL1_180'][1]
    res['b180']    = g['pL1_180'][2]

    res['maQSES']  = g['maQSES'][()]
    res['madSES']  = g['madSES'][()]

    h5.close()
    return res


res = read(files[0])
rL  = []
for i in range(len(args.inp)):
    try:
        rL.append( read(files[i]) )
    except:
        print "%s failed" % files[i]
        pass

rL = np.hstack(rL)


tpk = qalg.rec2tab(rL)
tpk.table_name = 'pk'
tpk.write('sqlite',args.out,overwrite=True)
