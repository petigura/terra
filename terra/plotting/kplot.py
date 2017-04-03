"""
Plotting methods for the following classes

- Lightcurve
- Grid 
- Peak

I define these functions in a separate module so the classes
themselves can be instantiated without importing matplotlib which is not
possible on some non-interactive platforms
"""
from scipy import ndimage as nd
import numpy as np
from numpy import ma
from matplotlib import mlab
from matplotlib.pylab import *
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import pandas as pd
import h5py

from .. import tval
from .. import config
from . import sketch

seasonColors = ['r','c','m','g']
tprop = dict(name='monospace')
bbox = dict(
    boxstyle="round", fc="w",alpha=.5,ec='none'
)
annkw = dict(
    xycoords='data',textcoords='offset points',bbox=bbox,weight='light'
)

def MC_diag(dv):
    fig = figure(figsize=(10,6))
    gs = GridSpec(8,5)

    axFitResid = fig.add_subplot(gs[0,0:2])
    axFitClean = fig.add_subplot(gs[1:4,0:2])
    axFitFull   = fig.add_subplot(gs[4:,0:2],sharex=axFitClean)

    plotfitchain(dv,axFitResid,axFitClean)

    sca(axFitResid)
    yticksppm()
    sca(axFitFull)
    plotPF(dv,0)

    for ax in [axFitResid,axFitClean,axFitFull]:
        sca(ax)
        yticksppm()
    for ax in [axFitResid,axFitClean]:
        ax.xaxis.set_visible(False)
    

    axbp = fig.add_subplot(gs[:4,2:4])
    sca(axbp)
    gca().xaxis.set_visible(False)
    plot__covar(dv)
    ylim(0,0.1)
    xlim(0,1)
    ytickspercen()
    
    axbpzoom = fig.add_subplot(gs[4:,2:4])
    sca(axbpzoom)
    plot_bp_covar(dv)
    ytickspercen()

    d  = tval.TM_getMCMCdict(dv)
    d  = tval.TM_unitsMCMCdict(d)
    sd = tval.TM_stringMCMCdict(d)

    s = """\
kic %(skic)09d
P   %(P).3f
t0  %(t0).2f

p   %(p)s +/- %(up)s %%
tau %(tau)s +/- %(utau)s hrs
b   %(b)s +/- %(ub)s
""" % sd


    text(1.1,0,s,transform=axbpzoom.transAxes,family='monospace',size='large')
    tight_layout()
    fig.subplots_adjust(hspace=0.0001)

def plotfitchain(dv,ax1,ax2):
    """
    Plot the median 10-min points 
    Plot the best fit
    Show the range of fits
    """
    fitgrp = dv['fit']
    t      = fitgrp['t'][:]
    f      = fitgrp['f'][:]
    fit    = fitgrp['fit'][:]

    sca(ax1)
    plot(t,f-fit)

    sca(ax2) 
    plot(t,f,'o',ms=3,mew=0)
    plot(t,fit,'r--',lw=3,alpha=0.4)
    # Show the range of fit from MCMC

    yL = dv['fit/fits'][:]
    lo,up  = np.percentile(yL,[15,85],axis=0)
    fill_between(t, lo, up, where=up>=lo, facecolor='r',alpha=.5,lw=0)

def plot_bp_covar(dv):
    """
    Plot the covariance between impact parameter, b, and radius ratio, p.
    """
    chain = dv['fit/chain'][:]
    uncert = dv['fit/uncert'][:]

    plot( abs(chain['b']), chain['p'], 'o', mew=0, alpha=.2, ms=2)
    
#    ub = uncert['b']
#    axvspan(ub[0],ub[1],alpha=.5)

    up = uncert['p']
    axhspan(up[0],up[2],alpha=.2,ec='none')

    xlabel('b')
    ylabel('Rp/Rstar %')

def yticksppm():
    """
    Convienence function to convert fractional flux to ppm.
    """
    yt = yticks()[0]
    yt = (yt*1e6).astype(int)
    yt = np.round(yt,-2)
    yticks(yt/1e6,yt)
    ylabel("flux (ppm)")

def ytickspercen():
    """
    Convienence function to convert fractional flux to ppm.
    """
    yt  = yticks()[0]
    yt  = np.round(yt*1e2).astype(int)
    syt = ["%i%%" %  i for i in yt]
    yticks(yt/1e2,syt)


def plotSeason(dv):
    cax = gca()
    rses = dv['SES']
    cax.plot(rses['season'],rses['ses']*1e6,'_',mfc='k',mec='k',ms=4,mew=2)
    cax.xaxis.set_visible(False)
    AddAnchored('Season SES',prop=tprop,  frameon=False,loc=3)
    xlim(-1,4)
    yl = ylim()
    ylim(-100,yl[1])

def plotScar(dv):
    """
    Plot scar plot

    Parameters
    ----------
    r : res record array containing s2n,Pcad,t0cad, column
    
    """
    r,lc = dv.RES,dv.lc
    bcut = r['s2n']> np.percentile(r['s2n'],90)
    x = r['Pcad'][bcut]
    x -= min(x)
    x /= max(x)
    y = (r['t0cad']/r['Pcad'])[bcut]
    plot(x,y,',',mew=0)
    gca().xaxis.set_visible(False)
    gca().yaxis.set_visible(False)

def plotCDF(dv):
    lc = dv['/pp/mqcal'][:]
    sig = nd.median_filter(np.abs(lc['dM3']),200)
    plot(np.sort(sig))
    sig = nd.median_filter(np.abs(lc['dM6']),200)
    plot(np.sort(sig))
    sig = nd.median_filter(np.abs(lc['dM12']),200)
    plot(np.sort(sig))
    gca().xaxis.set_visible(False)
    gca().yaxis.set_visible(False)

def plotMed(dv):
    lc = dv.lc
    t = lc['t']
    fmed = ma.masked_array(dv['fmed'][:],lc['fmask'])
    P = dv.attrs['P']
    t0 = dv.attrs['t0']
    df = dv.attrs['df']
    stack(t,fmed*1e6,P,t0,step=5*df)
    autoscale(tight=True)


def plotManyTrans(dv):
    PF = dv['lcPF0'][:]
    numTrans = (PF['t'] / dv.attrs['P']).astype(int) # number of the transit
    iTrans   = np.digitize(numTrans,np.unique(numTrans))
    xL,yL,qL = [],[],[]

    for i in np.unique(iTrans):
        xL.append( PF['tPF'][iTrans==i] )
        yL.append( PF['f'][iTrans==i] ) 
        qL.append( PF['qarr'][iTrans==i] ) 

    nTrans = len(xL)
    kw = {}
    kw['wData'] = np.nanmax([x.ptp() for x in xL]) * 1.1
    kw['hData'] = dv.attrs['df'] * 1e-6

    mad = np.median(np.abs(np.hstack(yL)))
    hStepData = 6 * mad
    if hStepData < kw['hData']:
        hStepData = kw['hData'] * 0.5
    
    kw['hStepData'] = hStepData
    kw['hAx'] = 0.04
    kw['wAx'] = 0.1

    kw1 = sketch.gridTraceSetup(**kw)
    xLout,yLout = sketch.gridTrace(xL,yL,**kw1)

    nTransMax = kw1['nCols']*kw1['nRows']
    if nTransMax < nTrans:
        print "Too many transits, rebin"
        bfac = int(nTrans / nTransMax) + 1
        xL2,yL2 = rebin(xL,yL,bfac)
        xLout,yLout = sketch.gridTrace(xL2,yL2,**kw1)

    if nTransMax > 2 * nTrans:
        print "Extra space"
        kw['hStepData'] *= 1.0*nTransMax / nTrans
        kw1 = sketch.gridTraceSetup(**kw)
        xLout,yLout = sketch.gridTrace(xL,yL,**kw1)

    for i in range(len(xLout)):
        q = int(min(qL[i]))
        season = q % 4
        if i % 2==0:
            color='k'
        else:
            color  = seasonColors[season]

        plot(xLout[i],yLout[i],color=color)
        
    autoscale(tight=True)
    xl = xlim() 
    if xl[1] < 1:
        xlim(xl[0],1)
    xl = xlim()
    yl = ylim()
    pad = 0.02
    xlim(xl[0]-3*pad,xl[1]+pad)
    ylim(yl[0]-pad,yl[1]+pad)

    add_scalebar(gca(),loc=3,matchx=False,matchy=False,sizex=1*kw1['t2ax'],sizey=1e-3*kw1['f2ax'],labelx='1 Day',labely='1000 ppm')    

def rebin(xL,yL,bfac):
    iStart = 0 
    iStop = iStart+bfac
    xL2,yL2 = [],[]
    while iStop < len(xL):
        x = np.hstack(xL[iStart:iStop])
        y = np.hstack(yL[iStart:iStop])

        sid = np.argsort(x)
        x = x[sid]
        y = y[sid]

        bins = np.linspace(x[0],x[-1]+0.0001,int(x.ptp()/config.lc) + 1)

        bc        = 0.5*(bins[1:]+bins[:-1])
        tot,b     = np.histogram(x,weights=y,bins=bins)
        count,b   = np.histogram(x,bins=bins)
        xL2.append(bc)
        yL2.append(tot/count)
        iStop+=bfac
        iStart+=bfac
    return xL2,yL2

def qstackplot(dv,func):
    """

    """
    qL = [int(i[0][1:]) for i in dv['/raw'].items()]

    qmi,qma = min(qL),max(qL)
    for i in range(qmi,qma+1):
        if i==qmi : 
            label = True
        else:
            label = False

        if qL.count(i)==1:
            func(dv,i,label)

def helper(i):
    season = (i+1) % 4    
    return dict( season = season,
                 year   = (i - season)/4 ,
                 qstr   = 'Q%i' % i )


def addtrans(raw,y,label,plot):
    """
    Add little triangles marking where the transits are.
    """
    if raw.dtype.names.count('finj')==1:
        finjkw = dict(color='m',mew=2,marker=7,ms=5,lw=0,mfc='none')
        if label==True:
            finjkw['label'] = 'Injected Transits'

        ym =  ma.masked_array(y, raw['finj'] > -1e-6 ) 
        t  =  raw['t']
        sL = ma.clump_unmasked(ym)
        id = [s.start for s in  ma.clump_unmasked(ym)]
        plot(t[id] ,ym[id] + 0.001,**finjkw)


def plotraw(dv):
    """
    """

    qlc = dv['/pp/cal'][:]
    t   = qlc['t']
    fm = ma.masked_array(qlc['f'],qlc['fmask'])
    plot(t,fm,label='Raw Photometry')

def plotcal(dv):
    qlc = dv['/pp/cal'][:]
    t   = qlc['t']

    fdt  = ma.masked_array(qlc['fdt'],qlc['fmask'])
    fit  = ma.masked_array(qlc['fit'],qlc['fmask'])
    fcal = ma.masked_array(qlc['fcal'],qlc['fmask'])

    plot(t ,fit, color='Tomato',label='Mode Calibration')
    plot(t ,fcal -0.001,color='k',label='Calibrated Photometry')
    legend()

def plotdt(dv):
    qlc = dv['/pp/cal'][:]
    t   = qlc['t']
    fdt = ma.masked_array(qlc['fdt'],qlc['fmask'])
    plot(t,fdt,label='Detrended Photometry')


def plot_lc(dv):
    fig,axL = subplots(nrows=2,figsize=(20,12),sharex=True)

    sca(axL[0])
    plotraw(dv)
    legend()

    sca(axL[1])
    plotdt(dv)
    plotcal(dv)
    legend()
    setp(fig,tight_layout=True)


#############################################################################

# -*- coding: utf-8 -*-
# -*- mode: python -*-
# Adapted from mpl_toolkits.axes_grid2
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)
 
from matplotlib.offsetbox import AnchoredOffsetbox
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
 
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, fc="none"))
 
        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                            align="center", pad=0, sep=sep)
 
        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)
 
def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes
 
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
 
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
 
    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)
 
    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)
 
    return sb

def wrapHelp(dv, x, ym, d, **kwargs):
    scale = 1e6
    pad =  dv.mean*3*scale

    stackkw = dict(step=pad,P=dv.P,t0=dv.t0)
    stackkw = dict(stackkw,**kwargs)
    df = stack(x,ym,**stackkw)
    pltkw = dict(alpha=0.2)
    df = stack(x,ym.data,pltkw=pltkw,**stackkw)    

    autoscale(tight=True)
    y0 = min(df['yshft']) - pad
    y1 = max(df['yshft']) + pad 
    ylim(y0,y1)



def plot_lc(pipe):
    lc = pipe.lc

    isOutlier = np.array(lc['isOutlier'])
    fig,axL = plt.subplots(nrows=2,figsize=(20,8),sharex=True)
    plt.sca(axL[0])

    f = ma.masked_array(lc['f'],lc['fmask'],fill_value=0)

    plt.plot(lc['t'],f.data,label='Full light curve')
    plt.plot(lc['t'],f,color='RoyalBlue',
             label='Masked light curve')
    plt.plot(
        lc['t'][isOutlier],f.data[isOutlier],'or',mew=0,
        alpha=0.5,label='Outliers Identfied in time-domain'
        )
    plt.ylabel('f')
    plt.legend(loc='best')

    plt.sca(axL[1])
    plt.plot(lc['t'],f,label='Masked light curve')
    plt.xlabel('Time')
    plt.legend(loc='best')
    fig.set_tight_layout(True)

