"""
Plotting methods for the following classes

- Lightcurve
- Grid 
- Peak

I define these functions in a seperate module so the classes
themselves can be instanated with out import matplotlib which is not
possible on some non-interactive platforms
"""

from matplotlib.pylab import *
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

import terra
import tval
import keplerio
import sketch
import config
from scipy import ndimage as nd
import numpy as np
from numpy import ma
from matplotlib import mlab
import sys

import pandas as pd
pd.set_eng_float_format(accuracy=3,use_eng_prefix=True)

seasonColors = ['r','c','m','g']
tprop = dict(name='monospace')

bbox=dict(boxstyle="round", fc="w",alpha=.5)
annkw = dict(xycoords='data',textcoords='offset points',bbox=bbox)

rc('axes',color_cycle=['RoyalBlue','Tomato'])
rc('font',size=8)

def plot_diag(h5):
    """
    Print a 1-page diagnostic plot of a given h5.
    """
    fig = figure(figsize=(20,12))
    gs = GridSpec(8,10)

    axPeriodogram  = fig.add_subplot(gs[0,0:8])
    axStack        = fig.add_subplot(gs[2:8 ,0:8])
    axStackZoom    = fig.add_subplot(gs[2:8 ,8:])
    axPF           = fig.add_subplot(gs[1,0:2])
    axPF180        = fig.add_subplot(gs[1,2:4],sharex=axPF,sharey=axPF)

    axScar     = fig.add_subplot(gs[1,5])
    axSingSES  = fig.add_subplot(gs[0,-2])
    axSeason   = fig.add_subplot(gs[0,-1])
    axAutoCorr = fig.add_subplot(gs[1,6])
    axCDF      = fig.add_subplot(gs[1,4])

    h5.noPrintRE = '.*?file|climb|skic|.*?folder'
    h5.noDiagRE  = \
        '.*?file|climb|skic|KS.|Pcad|X2.|mean_cut|.*?180|.*?folder'
    h5.noDBRE    = 'climb'

    s = pd.Series(tval.flatten(h5,h5.noDiagRE))
    s = "%i\n%s" % (h5.attrs['skic'], s.__str__())
    bbkw = dict(visible=True,fc='white',alpha=0.5)
    axStack.text( 0.87, 0.01, s, name='monospace',bbox=bbkw, \
                      transform=axStack.transAxes)
    tight_layout()

    gcf().subplots_adjust(hspace=0.01,wspace=0.01)

    sca(axPeriodogram)
    plotPeriodogram(h5)

    sca(axPF)
    plotPF(h5,0,diag=True)

    sca(axPF180)
    plotPF(h5,180,diag=True)



    sca(axStack)
    plotSES(h5)

    sca(axScar)
    plotScar(h5)

    sca(axSingSES)
    plotSingSES(h5)

    sca(axAutoCorr)
    plotAutoCorr(h5)

    sca(axCDF)
    plotCDF(h5)

    sca(axStackZoom)
    plotCalWrap(h5)

    sca(axSeason)
    plotSeason(h5)

    sca(axSingSES)
    plotSingSES(h5)
    tight_layout()

def plotSingSES(h5):    
    cax = gca()
    rses = h5['SES']
    plot(rses['tnum'],rses['ses']*1e6,'.')
    cax.xaxis.set_visible(False)
    at = AnchoredText('Transit SES',prop=tprop, frameon=False,loc=2)
    cax.add_artist(at)
    xl = xlim()
    xlim(xl[0]-1,xl[-1]+1)

def plotSeason(h5):
    cax = gca()
    rses = h5['SES']
    cax.plot(rses['season'],rses['ses']*1e6,'.')
    cax.xaxis.set_visible(False)
    at = AnchoredText('Season SES',prop=tprop, frameon=False,loc=2)
    cax.add_artist(at)
    xlim(-1,4)

def plotScar(h5):
    """
    Plot scar plot

    Parameters
    ----------
    r : res record array containing s2n,Pcad,t0cad, column
    
    """
    r,lc = terra.get_reslc(h5)
    bcut = r['s2n']> np.percentile(r['s2n'],90)
    x = r['Pcad'][bcut]
    x -= min(x)
    x /= max(x)
    y = (r['t0cad']/r['Pcad'])[bcut]
    plot(x,y,',',mew=0)
    gca().xaxis.set_visible(False)
    gca().yaxis.set_visible(False)

def plotCDF(h5):
    lc = h5['/pp/mqcal'][:]
    sig = nd.median_filter(np.abs(lc['dM3']),200)
    plot(np.sort(sig))
    sig = nd.median_filter(np.abs(lc['dM6']),200)
    plot(np.sort(sig))
    sig = nd.median_filter(np.abs(lc['dM12']),200)
    plot(np.sort(sig))
    gca().xaxis.set_visible(False)
    gca().yaxis.set_visible(False)

def plotAutoCorr(pk):
    xlabel('Displacement')
    plot(pk['lag'],pk['corr'])
    gca().xaxis.set_visible(False)
    gca().yaxis.set_visible(False)

def plotPF(h5,ph,diag=False):
    PF = h5['lcPF%i' % ph]
    x  = PF['tPF']
    
    @handelKeyError
    def plot_phase_folded():
        plot(x,PF['f'],',',color='k')
        bPF = h5['blc30PF%i' % ph][:]
        t   = bPF['tb']
        f   = bPF['med']
        plot(t,f,'+',ms=5,mew=3,color='Chartreuse')

    @handelKeyError
    def plot_fit():
        fitgrp = h5['fit']
        t    = fitgrp['t']
        f    = fitgrp['f']
        ffit = fitgrp['fit'][:]
        plot(t,ffit,'r--',lw=3,alpha=0.7)

    plot_phase_folded()
    if ph==0:
        plot_fit()
        title='Phase Folded'
    else:
        title='Phase Folded + 180'

    depth = h5.attrs['mean']
    if depth < 1e-4:
        ylim(-5*depth,5*depth)

    if diag:
        cax = gca()
        cax.xaxis.set_visible(False)        
        at = AnchoredText(title,prop=tprop,frameon=True,loc=2)
        cax.add_artist(at)

    xlabel('Phase')

def handelKeyError(func):
    """
    Cut down on the number of try except statements
    """
    def handelProblems():
        try:
            func()
        except KeyError:
            print "%s: KeyError" %  func.__name__ 
    return handelProblems

def wrapHelp(h5,x,ym,d):
    df = h5['fit'].attrs['pL0'][0]**2
    d['step'] = 3*df*1e6
    d['P']    = h5.attrs['P']
    d['t0']   = h5.attrs['t0']

    stack(x,ym,**d)
    pltkw   = dict(alpha=0.2)
    stack(x,ym.data,pltkw=pltkw,**d)    
    autoscale(tight=True)

def plotCalWrap(h5):
    d = dict(time=True)

    res,lc = terra.get_reslc(h5)
    ym = ma.masked_array(lc['fcal']*1e6,lc['fmask'])
    wrapHelp(h5,lc['t'],ym,d)
    ylabel('SES (ppm)')
    xlim(-2,2)

    axvline(0, alpha=.1,lw=10,color='m',zorder=1)
    gca().yaxis.set_visible(False)

def plotSES(h5):
    d = dict(time=False)

    res,lc = terra.get_reslc(h5)
    fm = ma.masked_array(lc['dM6']*1e6,lc['fmask'])
    wrapHelp(h5,lc['t'],fm,d)
    ylabel('SES (ppm)')
    axvline(0, alpha=.1,lw=10,color='m',zorder=1)
    axvline(.5,alpha=.1,lw=10,color='m',zorder=1)

    if lc.dtype.names.count('finj')==1:
        finjkw = dict(color='m',mew=2,marker=7,ms=5,lw=0,mfc='none')
        ym =  ma.masked_array(fm.data, lc['finj'] < -1e-6 ) 
        t  =  lc['t']
        id = [s.start for s in  ma.clump_masked(ym)]
        stack(t[id] ,ym.data[id] + 100,pltkw=finjkw,**d)

def plotPeriodogram(h5):
    cax = gca()
    res,lc = terra.get_reslc(h5)
    x = res['Pcad']*config.lc
    y = res['s2n']

    semilogx()
    xt = [0.1,0.2,0.5,1,2,5,10,20,50,100,200,400]

    xt = [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9] + \
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] +\
         [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90] +\
         [  0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    xticks(xt,xt)
    at = AnchoredText('Periodogram',prop=tprop, frameon=True,loc=2)
    cax.add_artist(at)
    cax.xaxis.set_ticks_position('top')
    ylabel('MES')
    
    plot(x,y)
    id = np.argsort( np.abs(x - h5.attrs['P']) )[0]
    plot(x[id],y[id],'ro')
    autoscale(axis='x',tight=True)

def plotMed(h5):
    lc = h5.lc
    t = lc['t']
    fmed = ma.masked_array(h5['fmed'][:],lc['fmask'])
    P = h5.attrs['P']
    t0 = h5.attrs['t0']
    df = h5.attrs['df']
    stack(t,fmed*1e6,P,t0,step=5*df)
    autoscale(tight=True)

def morton(h5):
    """
    Print a 1-page diagnostic plot of a given h5.
    """
    P  = h5.attrs['P']
    t0 = h5.attrs['t0']
    df = h5.attrs['df']

    fig = figure(figsize=(20,12))
    gs = GridSpec(4,3)

    axPF0   = fig.add_subplot(gs[0,0])
    axPF180 = fig.add_subplot(gs[0,1],sharey=axPF0)
    axPFSea = fig.add_subplot(gs[0,2],sharey=axPF0)
    axStack    = fig.add_subplot(gs[1:,:])

    for ax,ph in zip([axPF0,axPF180],[0,180]):
        sca(ax)        

        PF  = h5['lcPF%i'%ph]
        bPF = h5['blc30PF%i' % ph ]

        plot(PF['tPF'],PF['f'],',',color='k')
        plot(bPF['tb'],bPF['med'],'o',mew=0,color='red')
        xl = max(np.abs(PF['tPF']))
        xlim(np.array([-1,1])*xl )

    cax = gca()
    cax.yaxis.set_visible(False)
    at = AnchoredText('Phase Folded LC + 180',prop=tprop,frameon=True,loc=2)
    cax.add_artist(at)

    sca(axPF0)

    cax = gca()
    at = AnchoredText('Phase Folded LC',prop=tprop,frameon=True,loc=2)
    cax.add_artist(at)
    xlabel('t - t0 (days)')
    ylabel('flux')
    ylim(-3*df*1e-6,2*df*1e-6)

    sca(axPFSea)
#    import pdb;pdb.set_trace()

    for season in range(4):
        try:
            PF = h5['PF_Season%i' % season ][:]
            plot(PF['t'],PF['fmed'],color=seasonColors[season],label='%i' % season)
        except:
            pass

    at = AnchoredText('Transit by Season',prop=tprop,frameon=True,loc=2)
    cax = gca()
    cax.legend(loc='lower right',title='Season')
    cax.yaxis.set_visible(False)
    cax.add_artist(at)
    sca(axStack)
    plotManyTrans(h5)
 
    gcf().subplots_adjust(hspace=0.21,wspace=0.05,left=0.05,right=0.99,bottom=0.05,top=0.99)

def plotManyTrans(h5):
    PF = h5['lcPF0'][:]
    numTrans = (PF['t'] / h5.attrs['P']).astype(int) # number of the transit
    iTrans   = np.digitize(numTrans,np.unique(numTrans))
    xL,yL,qL = [],[],[]

    for i in np.unique(iTrans):
        xL.append( PF['tPF'][iTrans==i] )
        yL.append( PF['f'][iTrans==i] ) 
        qL.append( PF['qarr'][iTrans==i] ) 

    nTrans = len(xL)
    kw = {}
    kw['wData'] = np.nanmax([x.ptp() for x in xL]) * 1.1
    kw['hData'] = h5.attrs['df'] * 1e-6

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

def qstackplot(h5,func):
    """

    """
    qL = [int(i[0][1:]) for i in h5['/raw'].items()]

    qmi,qma = min(qL),max(qL)
    for i in range(qmi,qma+1):
        if i==qmi : 
            label = True
        else:
            label = False

        if qL.count(i)==1:
            func(h5,i,label)

def helper(i):
    season = (i+1) % 4    
    return dict( season = season,
                 year   = (i - season)/4 ,
                 qstr   = 'Q%i' % i )

def plotraw(h5,i,label):
    colors = ['RoyalBlue','Black']
    d = helper(i)
    year,season = d['year'],d['season']
    xs =  365.25*year
    ys =  year*0.01

    qlc = h5['/raw/%(qstr)s' % d ][:]
    dt = h5['/pp/dt/%(qstr)s' % d ][:]

    t   = qlc['t']
    fm = ma.masked_array(qlc['f'],qlc['isBadReg'])
    ftnd = ma.masked_array(dt['ftnd'],qlc['fmask'])
    foutlier  = fm.copy()
    foutlier.mask = ~qlc['isOutlier']

    fspsd  = fm.copy()
    fspsd.mask = ~qlc['isStep']

    fkw     = dict(color=colors[year % 2],lw=3)
    foutkw  = dict(color=colors[year-1 % 2],lw=0,marker='x',mew=2,ms=5)
    fallkw  = dict(color=colors[year % 2],lw=3,alpha=0.4)
    ftndkw  = dict(color='Tomato',lw=2)
    fspsdkw = dict(color='Chartreuse',lw=10,alpha=0.5)

    if label==True:
        fkw['label']     = 'Raw Phot'
        fallkw['label']  = 'Removed by Hand'
        ftndkw['label']  = 'High Pass Filt'
        foutkw['label']  = 'Outlier'
        fspsdkw['label'] = 'SPSD'

    def plot(*args,**kwargs):
        plot( args[0] - xs,args[1] -ys, **kwargs )

    plot(t , fm        ,**fkw)
    plot(t , fm.data   ,**fallkw)
    plot(t , ftnd      ,**ftndkw)
    plot(t , foutlier  ,**foutkw)
    plot(t , fspsd     ,**fspsdkw)

    def plot(*args,**kwargs):
        plot( args[0] - xs,args[1] -ys+0.001, **kwargs )

    addtrans(qlc,fm.data,label,plot)

    xy =  (t[0]-xs , fm.compressed()[0]-ys)
    annotate(d['qstr'], xy=xy, xytext=(-10, 10), **annkw)




    legend(loc='upper left')

def addtrans(raw,y,label,plot):
    """
    Add little triangles marking where the transis are.
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

def plotcal(h5,i,label):
    colors = ['RoyalBlue','k']
    d = helper(i)
    year,season = d['year'],d['season']

    raw = h5['/raw/%(qstr)s'    % d][:]
    dt  = h5['/pp/dt/%(qstr)s'  % d][:]
    cal = h5['/pp/cal/%(qstr)s' % d][:]

    t = raw['t']

    fdt  = ma.masked_array(dt['fdt'],raw['fmask'])
    fit  = ma.masked_array(cal['fit'],raw['fmask'])
    fcal = ma.masked_array(cal['fcal'],raw['fmask'])

    xs =  365.25*year
    ys =  year*0.003 

    def plot(*args,**kwargs):
        plot( args[0] - xs,args[1] -ys, **kwargs )


    plot(t ,fit, color='Tomato')
    plot(t ,fcal -0.001,color=colors[i%2])

def plotdt(h5,i,label):
    colors = ['RoyalBlue','k']
    d = helper(i)
    year,season = d['year'],d['season']
    xs =  365.25*year
    ys =  year*0.003

    raw = h5['/raw/%(qstr)s'   % d][:]
    t   = raw['t']
    dt  = h5['/pp/dt/%(qstr)s' % d][:]

    fdt  = ma.masked_array(dt['fdt'],raw['fmask'])

    def plot(*args,**kwargs):
        plot( args[0] - xs,args[1] -ys, **kwargs )

    plot(t,fdt,color=colors[i%2])

def plot_lc(h5):
    fig,axL = subplots(nrows=2,figsize=(20,12),sharex=True)

    try:
        sca(axL[0])
        qstackplot(h5,plotraw)

        sca(axL[1])
        qstackplot(h5,plotdt)
        qstackplot(h5,plotcal)

    except:
        print sys.exc_info()[1]

    tight_layout()
    xlim(250,650)

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

def stack(t,y,P=0,t0=0,time=False,step=1e-3,maxhlines=50,pltkw={}):
    """
    Plot the SES    

    time : plot time since mid transit 
    
    set colors by changing the 'axes.colorcycle'
    """
    ax = gca()

    t = t.copy()

    # Shift t-series so first transit is at t = 0 
    dt = tval.t0shft(t,P,t0)
    t += dt
    phase = mod(t+P/4,P)/P-1./4

    # Associate each section of length P to a specific
    # transit. Sections start 1/4 P before and end 3/4 P after.
    label = np.floor(t/P+1./4).astype(int) 
    labelL = unique(label)

    xshft = 0
    yshft = 0
    for l in labelL:
        # Calculate shift in t-series.
        row,col = np.modf(1.*l/maxhlines)
        row = row*maxhlines
        xshift = col
        yshift = -row*step

        blabel = (l == label)
        phseg = phase[blabel]
        yseg  = y[blabel]
        sid   = np.argsort(phseg)
        phseg = phseg[sid]
        yseg  = yseg[sid]


        def plot(*args,**kwargs):
            ax.plot(args[0] +xshift, args[1] + yshift,**kwargs)

        if time:
            plot(phseg*P, yseg,**pltkw)
        else:
            plot(phseg, yseg, **pltkw)

        imin = np.argmin(np.abs(phseg+xshift))
        s = str(np.round(t[blabel][imin]-dt,decimals=1))
        ax.text(0,yshift,s)

    xlim(-.25,.75)
