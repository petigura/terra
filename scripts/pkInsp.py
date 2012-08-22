#!/usr/bin/env python
from argparse import ArgumentParser
from matplotlib.pylab import *
from matplotlib.gridspec import GridSpec
import h5py
import numpy as np
import atpy
from scipy.optimize import fmin

import sketch
import tfind
import keptoy
import tval
import string
import keplerio
from tval import readPkScalar

nbins = 20
prsr = ArgumentParser()

phelp = """
Parameters
if db is set: KIC, pknum 
else        : P (days), t0 (days), tdur (hours), Depth (ppm)'
"""
prsr.add_argument('p',nargs='+',type=str,help=phelp)
prsr.add_argument('cal' ,type=str,)
prsr.add_argument('grid',type=str,)
prsr.add_argument('pk'  ,type=str,)
prsr.add_argument('-o',type=str,default=None,help='png')
prsr.add_argument('--epoch',type=int,default=0,help='shift wrt fits epoch')
args  = prsr.parse_args()

cal  = args.cal
grid = args.grid
pk   = args.pk

info = readPkScalar(pk)
pk   = h5py.File(pk)
grp  = pk['/pk0']

t0 = info['t0']
#t0 += args.epoch
P  = info['P']

fig = figure(figsize=(18,10))
hgrd = h5py.File(grid,'r+') 
res = hgrd['RES']

hcal = h5py.File(cal,'r+') 
lc   = hcal['LIGHTCURVE']
fcal = ma.masked_array(lc['fcal'],lc['fmask'])
t    = lc['t']
tdur = info['tdur']
df   = info['df']

rec = tval.transLabel(t,P,t0,tdur)
tdurcad = int(np.round(tdur / keptoy.lc))
dM = tfind.mtd(t,fcal,tdurcad)

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
plotPF(grp['lcPF180'][:])
gca().xaxis.set_visible(False)
gca().yaxis.set_visible(False)

sca(axPF)
plotPF(grp['lcPF'][:])
gca().xaxis.set_visible(False)
gca().yaxis.set_visible(False)
ylim(-5*df,3*df)

sca(axStack)
plotSES()

sca(axScar)
sketch.scar(res)


qrec = keplerio.qStartStop()
q = np.zeros(t.size) - 1
for r in qrec:
    b = (t > r['tstart']) & (t < r['tstop'])
    q[b] = r['q']

tLbl = rec['tLbl']
b = tLbl >= 0

tnum = tLbl[b]
ses  = dM[b]
q    = q[b]
season = mod(q,4)
axSeason.set_xlim(-1,4)


axSES.plot(tnum,ses)
axSeason.plot(season,ses,'.')


def p1elrec(rec):
    sout = ''
    for n in rec.dtype.names:
        v = rec[0][n]
        if is_numlike(v):
            sout += string.ljust(n,10) + string.ljust('  %.3g' %  v,6)
        elif is_string_like(v):
            sout += v
        sout +='\n'
    return sout

gcf().text( 0.88, 0.05, p1elrec(info), size=12, name='monospace',
            bbox=dict(visible=True,fc='white'))

tight_layout()
gcf().subplots_adjust(hspace=0.01,wspace=0.01)
if args.o is not None:
    fig.savefig(args.o)
else:
    show()
    
