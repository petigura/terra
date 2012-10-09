from argparse import ArgumentParser
from numpy import ma
import keptoy
import h5py
import h5plus
import numpy as np
import tfind
import cotrend
import config

P1 = int(np.floor(config.P1/keptoy.lc))
P2 = int(np.floor(config.P2/keptoy.lc))

prsr = ArgumentParser(description='Run grid search')
prsr.add_argument('inp',type=str,help='input file')
prsr.add_argument('out',type=str,help='output file')
args = prsr.parse_args()

h5    = h5py.File(args.inp,'r+')
h5out = h5plus.File(args.out)

lc  = h5['mqcal'][:]

fm  = ma.masked_array(lc['fcal'],lc['fmask'],fill_value=0)
rtd = tfind.tdpep(lc['t'],fm,P1,P2,config.twdG)
r   = tfind.tdmarg(rtd)
h5out['RES'] = r
print "grid: Created %s" % h5out
h5out.close()
