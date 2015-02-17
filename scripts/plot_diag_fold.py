#!/usr/bin/env python

#
# Create a diagnostic plot based on the Kepler team KOI ephemeris
#

import matplotlib
from argparse import ArgumentParser

prsr = ArgumentParser()

prsr.add_argument('koi'  ,type=str,)
prsr.add_argument('-o',type=str,default=None,help='png')
args  = prsr.parse_args()
koi = args.koi

if args.o:
    matplotlib.use('Agg')

import h5py
import kplot
import os
import h5plus
import analysis
import keptoy
import numpy as np
from matplotlib.pylab import plt

q12 = analysis.loadKOI('Q12')
q12 = q12.dropna() 
q12.index =q12.koi
q12['Pcad']  = q12['P'] / keptoy.lc 

q12['twd']   = np.round(q12['tdur'] / 24 / keptoy.lc ).astype(int)
q12['mean']  = q12['df'] * 1e-6
q12['s2n']   = 0 
q12['noise'] = 0 

tpar = dict(q12.ix[koi]) # transit parameters for a particular koi

opath = 'grid/%(kic)09d.h5' % tpar
npath = opath.split('/')[-1].split('.')[0]+'_temp.h5'
cmd  = 'cp %s %s' % (opath,npath) 
os.system(cmd)

with h5plus.File(npath) as h5:
    kplot.plot_diag(h5,tpar=tpar)

    if args.o is not None:
        plt.gcf().savefig(args.o)
        print "created figure %s" % args.o
    else:
        plt.show()
    
