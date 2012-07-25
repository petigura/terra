#!/usr/bin/env python

"""
Split h5 file into individual files
"""
from argparse import ArgumentParser
from matplotlib.pylab import *
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import sqlite3
import h5py
import numpy as np

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

prsr.add_argument('p',nargs='+',type=float,help=phelp)
prsr.add_argument('--db',type=str,help='Database file')
prsr.add_argument('--cal',type=str,help='fcal file')
prsr.add_argument('--grid',type=str,help='fcal file')
prsr.add_argument('-o',type=str,default=None,help='png')
prsr.add_argument('--epoch',type=int,default=0,help='shift wrt fits epoch')
args  = prsr.parse_args()
if args.db !=None:
    assert len(args.p) == 2,'must specify KIC,pknum'
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cmd = """
SELECT P,epoch,twd*29.4/60.,mean*1e6,KIC
FROM pk 
WHERE 
KIC=%(KIC)i 
AND 
pknum=%(pknum)i;
""" % dict(KIC=args.p[0],pknum=args.p[1])
    cur.execute(cmd)
    P,t0,tdur,df,KIC = cur.fetchone()
else:
    P,t0,tdur,df = args.p
    KIC = 0

t0 += args.epoch

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


kicinfo = """
KIC  %(KIC)09d
P    %(P).2f
t0   %(t0).2f
Dur  %(tdur).2f
df   %(df).2f
""" % dict(P=P,t0=t0,tdur=tdur,df=df,KIC=KIC) 


def plotPF():
    # Plot phase folded LC
    tLbl,cLbl = tval.transLabel(t,P,t0,tdur/24)
    fldt = tval.LDT(t,fcal,cLbl,tLbl)
    tm = ma.masked_array(t,fldt.mask,copy=True)
    dt = tval.t0shft(t,P,t0)
    tm += dt
    tfold,ffold = mod(tm[~tm.mask]+P/2,P)-P/2,fldt[~fldt.mask] 
    bins=linspace(tfold.min(),tfold.max(),nbins)
    s,bins = histogram(tfold,weights=ffold,bins=bins)
    c,bins = histogram(tfold,bins=bins)

    tprop = dict(size=10,name='monospace')
    at = AnchoredText(kicinfo,prop=tprop,frameon=True,loc=3)

    sca(axPF)
    plot(tfold,ffold,',',alpha=.5)
    plot(bins[:-1]+0.5*(bins[1]-bins[0]),s/c,'o')
    gca().add_artist(at)
    axhline(0,alpha=.3)
    ylim( np.percentile( ffold,(5,95) ) )

def plotSES():
    # Plot SES
    tdurcad = int(np.round(tdur / 24. / keptoy.lc))
    dM = tfind.mtd(t,fcal,np.zeros(fcal.size).astype(bool),fcal.mask,tdurcad)
    sca(axStack)
    sketch.stack(t,dM,P,t0,step=1e-6*df)
    autoscale(tight=True)

def plotGrid():
    sca(axGrid)
    x = res['Pcad']*keptoy.lc
    plot(x,res['s2n'])
    id = np.argsort( np.abs(x - P) )[0]
    plot(x[id],res['s2n'][id],'ro')
    axGrid.xaxis.set_visible(False)
    sca(axPep)
    
    p90 = np.percentile(res['s2n'],90)
    bplot = res['s2n'] > p90
#    bplot = bplot.astype(float)
#    import pdb;pdb.set_trace()
    scatter(x[bplot],res['epoch'][bplot],edgecolor='none',
            c=res['s2n'][bplot],cmap=cm.bone_r,alpha=0.5)

    autoscale(tight=True)

gs = GridSpec(nrows,4)
if args.grid is not None:
    axGrid  = fig.add_subplot(gs[0,0:3])
    axPep   = fig.add_subplot(gs[1,0:3],sharex=axGrid)

    axStack = fig.add_subplot(gs[2: ,0:3])
    axPF    = fig.add_subplot(gs[0:2,3:4])

    plotPF()
    plotSES()
    plotGrid()
else:
    axPF    = fig.add_subplot(gs[0])
    axStack = fig.add_subplot(gs[1:])
    plotPF()
    plotSES()

tight_layout()

if args.o is not None:
    fig.savefig(args.o)
else:
    show()
    
