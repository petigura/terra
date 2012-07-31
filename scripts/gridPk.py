#!/usr/bin/env python
"""
Find the peaks in a grid.
"""
from argparse import ArgumentParser
import h5py
import h5plus
import tval
from matplotlib import mlab
import numpy as np
from numpy import ma


prsr = ArgumentParser(description='Find peaks in grid file')
prsr.add_argument('inp',type=str, help='input file')
prsr.add_argument('out',type=str, help='out file')
prsr.add_argument('--n',type=int, default=10,help='Number of peaks to store')
prsr.add_argument('--cal',type=str, help='lc file')

args  = prsr.parse_args()

print "inp: %s" % args.inp
print "out: %s" % args.out

hgd  = h5py.File(args.inp,'r+') 
res  = hgd['RES'][:]
rgpk = tval.gridPk(res)
rgpk = rgpk[-args.n:][::-1]  # Take the last n peaks

hpk = h5plus.File(args.out)
if args.cal is not None:
    hlc = h5py.File(args.cal,'r+')
    lc = hlc['LIGHTCURVE'][:]
    fm = ma.masked_array(lc['fcal'],lc['fmask'],fill_value=0)
    for i in range( rgpk.size ):
        # must work with length 1 array not single record
        r = rgpk[i:i+1] 
        Pcad0 = r['Pcad']


        rLarr = tval.harmSearch(res,lc['t'],fm,Pcad0)
        hpk.create_dataset('h%i' % i,data=rLarr)
        rgpk[i:i+1] = rLarr[np.nanargmax(rLarr['s2n'])]

rgpk = mlab.rec_append_fields(rgpk,'pknum',np.arange(rgpk.size) )
hpk.create_dataset('RES',data=rgpk)
for k in hgd.attrs.keys():
    hpk.attrs[k] = hgd.attrs[k]


hpk.close()
hgd.close()
