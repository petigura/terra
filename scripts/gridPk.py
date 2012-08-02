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
import keptoy
from scipy.optimize import fmin

prsr = ArgumentParser(description='Find peaks in grid file')
prsr.add_argument('inp',type=str, help='input file')
prsr.add_argument('out',type=str, help="""
output file.  If None, we replace .grid.h5 --> .pk.h """)
prsr.add_argument('--n',type=int, default=10,help='Number of peaks to store')
prsr.add_argument('--cal',type=str, help='lc file')

args  = prsr.parse_args()

out = h5plus.ext(args.inp,'.pk.h5',out=args.out)
print "inp: %s" % args.inp
print "out: %s" % out

hgd  = h5py.File(args.inp,'r+') 
res  = hgd['RES'][:]
rgpk = tval.gridPk(res)
rgpk = rgpk[-args.n:][::-1]  # Take the last n peaks
rgpk = mlab.rec_append_fields( rgpk,'scar',tval.scar(res) )


rgpk = mlab.rec_append_fields( rgpk,['tdf','tfdur','twdur'],[np.zeros(rgpk.size)]*3 )

hpk = h5plus.File(out)
if args.cal is not None:
    hlc = h5py.File(args.cal,'r+')
    lc = hlc['LIGHTCURVE'][:]
    fm = ma.masked_array(lc['fcal'],lc['fmask'],fill_value=0)

    t  = lc['t']
    for i in range( rgpk.size ):
        # must work with length 1 array not single record
        r = rgpk[i:i+1] 
        Pcad0 = r['Pcad']
        P = Pcad0*keptoy.lc       # days
        tdur = r['twd']*keptoy.lc # days
        t0 = r['t0']
        df = r['mean']
        

        rLarr = tval.harmSearch(res,t,fm,Pcad0)
        hpk.create_dataset('h%i' % i,data=rLarr)
        rgpk[i:i+1] = rLarr[np.nanargmax(rLarr['s2n'])]

        # Add in trapezoid fits
        x,y = tval.PF(t,fm,P,t0,tdur)
        y = ma.masked_invalid(y)
        x.mask = x.mask | y.mask
        x,y = x.compressed(),y.compressed()

        obj = lambda p : np.sum((y - keptoy.trap(p,x))**2)
        p0 = [1e6*df,tdur,.1*tdur]
        p1 = fmin(obj,p0,disp=1)

        rgpk[i:i+1]['tdf']    = p1[0] 
        rgpk[i:i+1]['tfdur']  = p1[1]
        rgpk[i:i+1]['twdur']  = p1[2]


pl = [50,90,99]
for p in pl:
    val = np.percentile(res['s2n'],p)
    rgpk = mlab.rec_append_fields( rgpk,'p%i' % p,val)

rgpk = mlab.rec_append_fields(rgpk,'pknum',np.arange(rgpk.size) )
hpk.create_dataset('RES',data=rgpk)
for k in hgd.attrs.keys():
    hpk.attrs[k] = hgd.attrs[k]

hpk.close()
hgd.close()
