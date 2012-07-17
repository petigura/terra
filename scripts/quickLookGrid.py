#!/usr/bin/env python

"""
Split h5 file into individual files
"""
from argparse import ArgumentParser
from matplotlib.pylab import *
from matplotlib.gridspec import GridSpec
import h5py
import keptoy
prsr = ArgumentParser()
prsr.add_argument('inp',type=str, help='input file')
prsr.add_argument('out',type=str, help='out file')

args  = prsr.parse_args()
nsteps = 10
h5 = h5py.File(args.inp) 
res = h5['RES']

gs = GridSpec(4,1)
fig = figure(figsize=(18,10))

axSm = fig.add_subplot(gs[0]) 
axBg = fig.add_subplot(gs[1:]) 

P = res['Pcad']*keptoy.lc
s2n = res['s2n']

axSm.plot(P,s2n)
xshft = P.ptp()/nsteps
yshft = np.percentile(s2n,99) - np.percentile(s2n,1)


rcParams.update({'axes.color_cycle':['b','r']})

for i in range(nsteps):
    axBg.plot(P-xshft*i,s2n-yshft*i)
    
xlim(P[0],P[0]+xshft) 
tight_layout()

fig.savefig(args.out)
