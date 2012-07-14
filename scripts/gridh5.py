import sim
import argparse
import atpy
from numpy import ma
import keptoy
import h5py
import h5plus
import numpy as np
import tfind

parser = argparse.ArgumentParser(description='Run grid search')

parser.add_argument('inp',  type=str   , help='input file')
parser.add_argument('out',  type=str   , help='output file')
parser.add_argument('--n',  type=int,default=0,   help= 'number of searches to run.  If this keyword is set, we search for only on the first n rows.')

args = parser.parse_args()
h5 = h5py.File(args.inp)

dslc = h5['LIGHTCURVE']
nstars,ncad = dslc.shape
if args.n is not 0:
    nstars = args.n
    
print "running grid search on %i stars" % nstars
rL = []
for i in range(nstars):    
    lc = dslc[i]
    t = lc['t']
    fcal = ma.masked_array(lc['fcal'],lc['fmask'],fill_value=0)

    isStep = np.zeros(fcal.size).astype(bool)

    P1 = int(np.floor(5./keptoy.lc))
    P2 = int(np.floor(50./keptoy.lc))
    twdG = [3,5,7,10,14,18]

    rtd = tfind.tdpep(t,fcal,isStep,P1,P2,twdG)
    r   = tfind.tdmarg(rtd)
    rL.append(r)

rL = np.vstack(rL)

h5 = h5plus.File(args.out)
h5.create_dataset('RES',data=rL)
print "grid: Created %s" % args.out
