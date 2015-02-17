#!/usr/bin/env python
from argparse import ArgumentParser
from matplotlib.pylab import *
from matplotlib.gridspec import GridSpec
import h5py
import keptoy
import numpy as np

nsteps = 10

prsr = ArgumentParser()
prsr.add_argument('inp',type=str, help='input file')
prsr.add_argument('out',type=str, help='out file')
prsr.add_argument('--kic',type=int,help='If h5 file contains more than one lc, specify it with the KIC')
args  = prsr.parse_args()

print "inp: %s" % args.inp

hcal = h5py.File(args.inp)
if args.kic is not None:
    idkic = np.where(hcal['KIC'][:]==args.kic)[0][0]
    print "%09d is at %i" % (args.kic,idkic)
    lc   = hcal['LIGHTCURVE'][idkic]
else:
    lc   = hcal['LIGHTCURVE']

rcParams.update({'axes.color_cycle':['b','r']})
gs  = GridSpec(4,1)
fig = figure(figsize=(18,10))

axSm = fig.add_subplot(gs[0]) 
axBg = fig.add_subplot(gs[1:])

axSm.plot(lc['t'],lc['fcal'])

xBg,yBg = lc['t'],lc['fcal']
xBg = ma.masked_invalid(xBg)
yBg = ma.masked_invalid(yBg)

xshft = int(xBg.compressed().ptp())/nsteps
yshft = np.percentile(yBg.compressed(),99) - np.percentile(yBg.compressed(),1)

for i in range(nsteps):
    axBg.plot(xBg-xshft*i,yBg-yshft*i)
xlim(xBg[0],xBg[0]+xshft) 

sca(axBg)
xlabel('Time (days)')
ylabel(r'$\delta F$ (ppt)')

yt = yticks()[0]
yticks(yt,(np.round(yt*1e3,decimals=1)))

draw()
tight_layout()
fig.savefig(args.out)
print "out: %s" % args.out
