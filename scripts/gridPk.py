#!/usr/bin/env python
"""
Find the peaks in a grid.
"""
from argparse import ArgumentParser
import h5py
import h5plus
import tval
import numpy as np
import keptoy
import sqlite3
import os

prsr = ArgumentParser(description='Find peaks in grid file')
prsr.add_argument('grid',type=str, help=".grid.h5 file")
prsr.add_argument('cal',type=str, help=".cal.h5 file")
prsr.add_argument('db',type=str, help='Database with limb-darkening coeffs')
prsr.add_argument('out',type=str,nargs='?', help="output file.  If None, we replace .grid.h5 --> .pk.h5 ")
prsr.add_argument('--n',type=int, default=1,help='Number of peaks to store')
args  = prsr.parse_args()

out = h5plus.ext(args.grid,'.pk.h5',out=args.out)
print "grid: %s" % args.grid
print "out: %s" % out

hgd  = h5py.File(args.grid,'r+') 
res  = hgd['RES'][:]
rgpk = tval.gridPk(res)
rgpk = rgpk[-args.n:][::-1]  # Take the last n peaks

hlc = h5py.File(args.cal,'r+')
lc = hlc['LIGHTCURVE'][:]

hpk = h5plus.File(out)
skic = os.path.basename(args.grid).split('.')[0]

# Pull limb darkening coeff from database
con = sqlite3.connect(args.db)
cur = con.cursor()
cmd = "select a1,a2,a3,a4 from b10k where skic='%s' " % skic
cur.execute(cmd)
res = cur.fetchall()
climb = np.array(res[0])

for i in range(args.n):
    rpk = rgpk[i:i+1]
    rpk = dict(t0=rpk['t0'][i],P=rpk['Pcad'][i]*keptoy.lc,tdur=rpk['twd'][i]*keptoy.lc,df=rpk['mean'][i])
    out = tval.pkInfo(lc,res,rpk,climb)

    grp = hpk.create_group('pk%i' % i)
    for k in out.keys():
        grp.create_dataset(k,data=out[k])


hgd.close()
hlc.close()
hpk.close()
