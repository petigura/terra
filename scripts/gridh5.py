from argparse import ArgumentParser
from numpy import ma
import keptoy
import h5py
import h5plus
import numpy as np
import tfind
import cotrend
from config import twdG

P1 = int(np.floor(5./keptoy.lc))
P2 = int(np.floor(50./keptoy.lc))

prsr = ArgumentParser(description='Run grid search')
prsr.add_argument('inp',type=str,help='input file')
prsr.add_argument('out',nargs='?',type=str,help="""
output file. If none specified .cal.h5 -> .grid.h5""")
prsr.add_argument('--flux',type=str,default='fcal',help= 'Name of flux field to process.  Default is cal')
args = prsr.parse_args()

h5     = h5py.File(args.inp,'r+')
out    = h5plus.ext(args.inp,'.grid.h5',out=args.out)

lc   = h5['LIGHTCURVE'][:]
t = lc['t']

fm  = ma.masked_array(lc[args.flux],lc['fmask'],fill_value=0)
rtd = tfind.tdpep(t,fm,P1,P2,twdG)
r   = tfind.tdmarg(rtd)

h5out = h5plus.File(out)
h5out.create_dataset('RES',data=r)
h5out.close()
print "grid: Created %s" % out
