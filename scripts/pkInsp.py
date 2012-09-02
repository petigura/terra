#!/usr/bin/env python
from argparse import ArgumentParser

import h5py
import numpy as np
from numpy import ma

import atpy
from scipy.optimize import fmin
import sketch
import tfind
import keptoy
import tval
import string
import keplerio

### Helper Functions ###

def plotPF(PF):
    # Plot phase folded LC
    x,y,yfit = PF['tPF'],PF['fPF'],PF['fit']
    bv = ~np.isnan(x) & ~np.isnan(y)
    try:
        assert x[bv].size==x.size
    except AssertionError:
        print 'nans in the arrays.  Removing them.'
        x = x[bv]
        y = y[bv]
        yfit = yfit[bv]

    bins   = linspace(x.min(),x.max(),nbins)
    s,bins = histogram(x,weights=y,bins=bins)
    c,bins = histogram(x,bins=bins)
    plot(x,y,',',alpha=.5)
    plot(x,yfit,alpha=.5)        

    plot(bins[:-1]+0.5*(bins[1]-bins[0]),s/c,'o',mew=0)
    axhline(0,alpha=.3)

def plotSES():
    sca(axStack)
    sketch.stack(t,dM,P,t0,step=df)
    autoscale(tight=True)

def plotGrid():
    x = res['Pcad']*keptoy.lc
    plot(x,res['s2n'])
    id = np.argsort( np.abs(x - P) )[0]
    plot(x[id],res['s2n'][id],'ro')
    autoscale(tight=True)


###########################

nbins = 20
prsr = ArgumentParser()

phelp = """
Parameters
pk file
"""
prsr.add_argument('pk'  ,type=str,)
prsr.add_argument('-o',type=str,default=None,help='png')
prsr.add_argument('--epoch',type=int,default=0,help='shift wrt fits epoch')
args  = prsr.parse_args()

### Load up the data
pk   = args.pk
pk   = tval.Peak(pk)
t0 = pk.attrs['t0']
P  = pk.attrs['P']

hgrd = h5py.File(pk.attrs['gridfile'],'r+') 
res = hgrd['RES']

hcal = h5py.File(pk.attrs['mqcalfile'],'r+') 
lc   = hcal['LIGHTCURVE']
fcal = ma.masked_array(lc['fcal'],lc['fmask'])
t    = lc['t']
tdur = pk.attrs['tdur']
df   = pk.attrs['pL0'][0]**2

rec = tval.transLabel(t,P,t0,tdur*2)
tdurcad = int(np.round(tdur / keptoy.lc))
dM = tfind.mtd(t,fcal,tdurcad)
qrec = keplerio.qStartStop()
q = np.zeros(t.size) - 1
for r in qrec:
    b = (t > r['tstart']) & (t < r['tstop'])
    q[b] = r['q']

tRegLbl = rec['tRegLbl']

btmid = np.zeros(tRegLbl.size,dtype=bool) # True if transit mid point.
for iTrans in range(0,tRegLbl.max()+1):
    tRegSES = dM[ tRegLbl==iTrans ] 
    if tRegSES.count() > 0:
        maSES = np.nanmax( tRegSES.compressed() )
        btmid[(dM==maSES) & (tRegLbl==iTrans)] = True

tnum = tRegLbl[btmid]
ses  = dM[btmid]
q    = q[btmid]
season = np.mod(q,4)

### Now plot ###
# Whether or not we save determines which backend we want to use.

import matplotlib
from matplotlib.gridspec import GridSpec
if args.o:
    matplotlib.use('Agg')
from matplotlib.pylab import *

fig = figure(figsize=(18,10))
gs = GridSpec(5,10)
axGrid  = fig.add_subplot(gs[0,0:8])
axStack = fig.add_subplot(gs[2: ,0:8])
axPF    = fig.add_subplot(gs[1,0:4])
axPF180 = fig.add_subplot(gs[1,4:8],sharex=axPF,sharey=axPF)
axScar  = fig.add_subplot(gs[0,-1])
axSES   = fig.add_subplot(gs[1,-1])
axSeason= fig.add_subplot(gs[2,-1])

sca(axGrid)
plotGrid()

sca(axPF180)
plotPF(pk.ds['lcPF180'])
gca().xaxis.set_visible(False)
gca().yaxis.set_visible(False)

sca(axPF)
plotPF(pk.ds['lcPF0'])
gca().xaxis.set_visible(False)
gca().yaxis.set_visible(False)
ylim(-5*df,3*df)

sca(axStack)
plotSES()

sca(axScar)
sketch.scar(res)
gca().xaxis.set_visible(False)
gca().yaxis.set_visible(False)


axSeason.set_xlim(-1,4)
axSES.plot(tnum,ses*1e6,'.')
axSES.xaxis.set_visible(False)
axSeason.plot(season,ses*1e6,'.')
axSeason.xaxis.set_visible(False)

gcf().text( 0.88, 0.05, pk.__str__() , size=12, name='monospace',
            bbox=dict(visible=True,fc='white'))

tight_layout()
gcf().subplots_adjust(hspace=0.01,wspace=0.01)
if args.o is not None:
    fig.savefig(args.o)
else:
    show()
    
