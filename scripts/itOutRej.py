"""
Iterative Outlier Rejection
    
If a particular cadence is over-represented in the MES
periodogram, it is probably due to a single point excursion.

Parameters
----------

lc  : light curve record array
res : grid record array.
"""
from argparse import ArgumentParser
from numpy import ma
from keptoy import lc
import h5py
import h5plus
import numpy as np
import tfind
import cotrend
from scipy import ndimage as nd
import os

twdG = [3,5,7,10,14,18]
P1 = int(np.floor(5./lc))
P2 = int(np.floor(50./lc))
cut = 5e3


prsr = ArgumentParser(description='Run grid search')
prsr.add_argument('grid',type=str,help='input file')
prsr.add_argument('lc',type=str,help='lightcurve file')
prsr.add_argument('out',type=str,help='input file')

args = prsr.parse_args()

hlc   = h5py.File(args.lc,'r+')
hgd   = h5py.File(args.grid,'r+')

bn = os.path.basename(args.grid)

res0 = hgd['RES'][:]
lc0  = hlc['LIGHTCURVE'][:]

h5out = h5plus.File(args.out)

h5out.create_dataset('RES',data=res0)

it = 0 
done = False
#import pdb;pdb.set_trace()

while done is False:
    cad = lc0['cad']
    c = tfind.cadCount(cad,res0)

    bout = c > cut
    nout = c[bout].size
    print "%s: it%i maxcount = %i, %i outliers" % (bn,it,max(c),nout)
    if (nout==0 ) or (it > 2):
        done = True
    else:
        lc1   = lc0.copy()

        # Update the mask
        bout = nd.convolve(bout.astype(float),np.ones(20) ) >  0
        lc1['fmask']  = lc1['fmask'] | bout

        fm  = ma.masked_array(lc1['fcal'],lc1['fmask'],fill_value=0)
        rtd  = tfind.tdpep(lc1['t'],fm,P1,P2,twdG)
        res1 = tfind.tdmarg(rtd)

        # Save most recent iteration to RES,LC 

        h5out['RES'][:] = res1

        gpname = 'it%i' % it

        grp = h5out.create_group(gpname)

        print "Storing Previous Iteration to Group %s" % gpname
        grp.create_dataset('RES',data=res0)
        grp.create_dataset('LIGHTCURVE',data=lc0)
        lc0  = lc1.copy()
        res0 = res1.copy()

    it +=1

hlc.close()
hgd.close()
h5out.close()
print "Updated %s" % args.grid
