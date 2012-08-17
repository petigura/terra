#!/usr/bin/env python

"""
Split h5 file into individual files
"""
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
nbins = 20
prsr = ArgumentParser()

phelp = """
Parameters
if db is set: KIC, pknum 
else        : P (days), t0 (days), tdur (hours), Depth (ppm)'
"""

prsr.add_argument('p',nargs='+',type=str,help=phelp)
prsr.add_argument('--db',type=str,help='Database file')
prsr.add_argument('--cal',type=str,help='fcal file')
prsr.add_argument('--grid',type=str,help='fcal file')
prsr.add_argument('-o',type=str,default=None,help='png')
prsr.add_argument('--epoch',type=int,default=0,help='shift wrt fits epoch')
args  = prsr.parse_args()
if args.db !=None:
    assert len(args.p) == 2,'must specify KIC,pknum'
#    query = "SELECT * from pk WHERE sKIC='%s' and pknum=%s" % (args.p[0],args.p[1])
    query = "SELECT * from pk WHERE sKIC='%s'" % (args.p[0])
    t = atpy.Table('sqlite',args.db,query=query)
    assert t.data.size==1,'must return a single column'

    info = {}
    for k in t.keys():
        info[k] = t[k][0]

else:
    P,t0,tdur,df = args.p
    info = dict(P=P,t0=t0,tdur=tdur,df=df,KIC=0)

t0 = info['t0']
t0 += args.epoch
P  = info['P']


fig = figure(figsize=(18,10))
if args.grid is not None:
    nrows = 5
    hgrd = h5py.File(args.grid,'r+') 
    res = hgrd['RES']
else:
    nrows = 4

hcal = h5py.File(args.cal,'r+') 
lc   = hcal['LIGHTCURVE']
fcal = ma.masked_array(lc['fcal'],lc['fmask'])
t    = lc['t']


#info['scar'] *= 1e6
#info['mean'] *= 1e6


#info['tdur']  = info['twd']*keptoy.lc*24
tdur = info['tdur']

df = info['df']
#import pdb;pdb.set_trace()
def plotPF(t,y,P,t0,tdur):
    # Plot phase folded LC
    x,y = tval.PF(t,y,P,t0,tdur)

    y = ma.masked_invalid(y)
    x.mask = x.mask | y.mask
    x,y = x.compressed(),y.compressed()

    bins   = linspace(x.min(),x.max(),nbins)
    s,bins = histogram(x,weights=y,bins=bins)
    c,bins = histogram(x,bins=bins)
    plot(x,y,',',alpha=.5)
    plot(bins[:-1]+0.5*(bins[1]-bins[0]),s/c,'o')
    axhline(0,alpha=.3)

#    obj = lambda p : np.sum((y - keptoy.trap(p,x))**2)
#    p0 = [1e6*df,tdur/24.,.1*tdur/24.]
#    p1 = fmin(obj,p0,disp=1)
#    xfit = linspace(x.min(),x.max(),1000)
#    yfit = keptoy.trap(p1,xfit)
#    plot(xfit,yfit)

def plotSES():
    # Plot SES
    tdurcad = int(np.round(tdur / keptoy.lc))
    dM = tfind.mtd(t,fcal,tdurcad)
    sca(axStack)
    sketch.stack(t,dM,P,t0,step=df)
    autoscale(tight=True)

def plotGrid():
    x = res['Pcad']*keptoy.lc
    plot(x,res['s2n'])
    id = np.argsort( np.abs(x - P) )[0]
    plot(x[id],res['s2n'][id],'ro')
    autoscale(tight=True)

gs = GridSpec(nrows,5)
if args.grid is not None:
    axGrid  = fig.add_subplot(gs[0,0:4])
    axStack = fig.add_subplot(gs[2: ,0:4])
    axPF    = fig.add_subplot(gs[1,0:2])
    axPF180 = fig.add_subplot(gs[1,2:4],sharex=axPF,sharey=axPF)
    axScar  = fig.add_subplot(gs[0:2,4])

    sca(axGrid)
    plotGrid()

    sca(axPF180)
    plotPF(t,fcal,P,t0+P/2,tdur)
    gca().xaxis.set_visible(False)
    gca().yaxis.set_visible(False)

    sca(axPF)
    plotPF(t,fcal,P,t0,tdur)
    ylim(-5*df,3*df)

    sca(axStack)
    plotSES()

    sca(axScar)
    sketch.scar(res)

else:
    axPF    = fig.add_subplot(gs[0])
    axStack = fig.add_subplot(gs[1:])
    plotPF()
    plotSES()

print info
#info['drat'] = info['tfdur'] / info['twdur']
kicinfo = """
KIC  %(sKIC)s
P    %(P).2f
t0   %(t0).2f
Dur  %(tdur).2f
df   %(df).2f
p50  %(p50).2f
p90  %(p90).2f
p99  %(p99).2f
""" % info



gcf().text(.8,.1,kicinfo,size=18,name='monospace',bbox=dict(visible=True,fc='none'))

tight_layout()
gcf().subplots_adjust(hspace=0.01,wspace=0.01)

if args.o is not None:
    fig.savefig(args.o)
else:
    show()
    
