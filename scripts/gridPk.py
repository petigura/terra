#!/usr/bin/env python
"""
Find the peaks in a grid.
"""
from argparse import ArgumentParser
import h5py
import h5plus
import tval
prsr = ArgumentParser()

prsr.add_argument('inp',type=str, help='input file')
prsr.add_argument('out',type=str, help='out file')
prsr.add_argument('--n',type=int, default=10,help='Number of peaks to store')
args  = prsr.parse_args()

print "inp: %s" % args.inp
print "out: %s" % args.out

hgd  = h5py.File(args.inp,'r+') 
rgpk = tval.gridPk(hgd['RES'][:])
rgpk = rgpk[-args.n:][::-1]  # Take the last n peaks

hpk = h5plus.File(args.out)
hpk.create_dataset('RES',data=rgpk)
for k in hgd.attrs.keys():
    hpk.attrs[k] = hgd.attrs[k]

hpk.close()
hgd.close()
