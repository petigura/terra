"""
A module for visualizing Kepler data.

The idea is to keep it rough.
"""
import glob
import copy

from numpy import *
from matplotlib.pylab import *
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from ..keptoy import *
from .. import keptoy
from .. import tfind
from .. import tval


def gridTraceSetup(wAx=0.2,wData=0.2,hAx=0.1,hData=2e-3,hStepData=1e-3):
    """

    """
    out = {}
    out['t2ax']   = wAx / wData
    f2ax          = hAx / hData
    nCols         = int(1 / wAx)
    nRows         = int((1 - f2ax*hData) / (f2ax*hStepData))
    out['wPad']   = (1 - nCols * wAx)/(nCols+1.)
    out['nPlots'] = nRows * nCols

    out['f2ax']  = f2ax
    out['nCols'] = nCols
    out['nRows'] = nRows
    out['wAx']   = wAx
    out['hStepData']  = hStepData
    return out

def gridTrace(xL,yL,**kw):
    """
    Transform list of data (must be roughly equally sized in absisca
    and ordiate) nto a plotable grid.

    xL     : list of X values
    yL     : list of Y values
    wAx    : Width of each column in axis units
    wData  : Width of each column in data units (used to transform data to axis)
    hAx    : Height of each row in axis units
    hData  : Corresponding height in data units
    hStepData : Spacing of traces in data units
    """
    t2ax   = kw['t2ax']
    f2ax   = kw['f2ax']
    nRows  = kw['nRows']
    wPad   = kw['wPad']
    nPlots = kw['nPlots']
    wAx    = kw['wAx']
    hStepData = kw['hStepData']
    i = 0 
    xLout = []
    yLout = []
    while (i < nPlots) & (i < len(xL)):
        x = xL[i]
        y = yL[i]

        col = i / nRows
        row = i - col*nRows

        x0 = col*wAx + wPad*(col+1)
        y0 = 1 - hStepData * f2ax * row

        ax_x = t2ax * x + x0
        ax_y = f2ax * y + y0

        xLout.append(ax_x)
        yLout.append(ax_y)

        i+=1

    return xLout,yLout



def PF(t,y,P,t0,tdur):
    """
    Plot Phase-folded light curve
    """

    dt = tval.t0shft(t,P,epoch)
    tm += dt
    
    clf()
    ax = gca()
    ax.plot(mod(tm[~tm.mask]+P/2,P)-P/2,fldt[~fldt.mask],'.')

def stackold(x,y,size,pad=0.1,axl=None,**kw):
    """

    """
    # How many regions
    npanel = int(ceil( x.ptp() / size))
    gs = GridSpec(npanel,1)
    gs.update(hspace=0.001)

    for i in range(npanel):
        if axl != None:
            ax = axl[i]

        offset = size*i
        ax = plt.subplot( gs[i])
        ax.plot(x,y,**kw)
        ax.set_xlim(offset-pad*size,offset+(1+pad)*size)
        ax.axvline(offset,ls='--',lw=1,label='Padded')
        ax.axvline(offset+size,ls='--',lw=1)
        ax.annotate('Offset = %i' % offset,xy=(.01,.1),
                    xycoords='axes fraction')

        xa = ax.get_xaxis()
        ya = ax.get_yaxis()

        if i == 0:
            xa.set_ticks_position('top')
            ax.legend(loc='upper left')
        else:
            xa.set_visible(False)
            ya.set_visible(False)


        if axl != None:
            axl[i] = ax

    if axl != None:
        return axl


def DM(dM,P):
    plt.clf()
    Pcad = int(round(P/lc))
    dMW = tfind.XWrap(dM,Pcad,fill_value=nan)
    dMW = ma.masked_invalid(dMW)
    dMW.fill_value=0

    nT = dMW.shape[0]
    ncad = dMW.shape[1]
    t = arange(ncad)*lc


    [plt.plot(t,dMW[i,:]+i*2e-4,aa=False) for i in range(nT)]    
    dMW = ma.masked_invalid(dMW)
    plt.plot(t,dMW.mean(axis=0)*sqrt(nT) - 5e-4,lw=3)


def FOM(t0,dM,P,step=None,**kwargs):
    """
    Plot the figure of merit

    """
    if step is None:
        step = np.nanmax(dM.data)
    Pcad = int(round(P/lc))

    dMW = tfind.XWrap(dM,Pcad,fill_value=np.nan)
    dMW = ma.masked_invalid(dMW)
    dMW.fill_value=np.nan

    res = tfind.ep(t0,dM,Pcad)
    fom = res['fom']
    color = ['black','red']
    ncolor = len(['black','red'])
    for i in range(dMW.shape[0]):
        x = ma.masked_array(res['epoch'],mask=dMW[i,:].mask).compressed()
        y = dMW[i,:].compressed()
        plt.plot(x,y+i*step,color=color[mod(i,ncolor)] ,)

    plot(res['epoch'],res['fom'] -step )
    return dMW


def FOMblock(t0,dM,P,**kwargs):
    nt = int(dM.size *keptoy.lc / P)
    tpb = 20
    nblock = nt / tpb
    print nblock
    Pcad = int(P / keptoy.lc)
    if nblock >2 :
        for i in range(nblock):
            dMp = dM[Pcad*i*tpb:Pcad*(i+1)*tpb]
            print dMp.size
            FOM(t0 + i *( P+0.5),dMp,P,**kwargs)
    else:
        FOM(t0,dM,P,**kwargs)


def window(tRES,tLC):
    PcadG = (tRES.PG[0]/keptoy.lc).astype(int)
    filled = tfind.isfilled(tLC.t,tLC.f,20)
    win = tval.window(filled,PcadG)
    plot(PcadG*keptoy.lc,win)
    xlabel('Period (days)')
    ylabel('Window')


def LDT(t,fm,p):
    """
    Visualize how the local detrender works
    """

    ax = gca()

    p1L,idL  = tval.LDT(t,fm,p,wd=2)

    twd = 2./lc
    step = 3
    for i in range(len(p1L)):
        id = idL[i]
        p1 = p1L[i]
        trend = keptoy.trend(p1[3:],t[id])
        ffit  = keptoy.P051T(p1,t[id])
        ho = np.mean(t[id]) 
        ho -= step*np.floor(ho/p['P'])
        vo = np.mean(fm[id])
        plot(t[id]-ho,fm[id]-vo,',')
        plot(t[id]-ho,ffit-vo,'r',lw=2)
        plot(t[id]-ho,trend-vo,'c',lw=2)

        color = rcParams['axes.prop_cycle'][mod(i,4)]

def tfit(tsim,tfit):
    plot(tset.RES.PG[0],tset.RES.ddd[1]/tset.RES.sss[1],'o')
    
def eta(tres,KIC):
    """
    Plot detection efficency as a function of depth for a given star.
    """
    PL = unique(tres.Pblock)
    for P in PL:
        dfL = unique(tres.df)
        fgL = []
        efgL = []
        for df in dfL:
            cut = (tres.KIC == KIC ) & (tres.Pblock == P) & (tres.df == df)
            tc = tres.where( cut    ) 
            tg = tres.where( cut & tres.bg   ) 
            tb = tres.where( cut & ~tres.bg  )            
            nc,ng,nb = tc.data.size,tg.data.size,tb.data.size

            print "%s %03d %7.5f %02d %02d %02d" % (KIC,P,df,ng,nb,nc)
            fgL.append(1.*ng/nc )
            efgL.append(1./sqrt(nc)  )
        errorbar(dfL,fgL,efgL,label='P-%03d' % P)

    xlabel(r'$\Delta F / F$')
    ylabel('Detection Efficiency')
    title('%d' % KIC)
    legend(loc='best')
    draw()

def markT(f,p,wd=2):
    P = p['P']
    epoch = p['epoch']
    tdur = p['tdur']

    twd      = round(tdur/lc)
    Pcad     = int(round(P/lc))
    epochcad = int(round(epoch/lc))
    wdcad    = int(round(wd/lc))

    f0W = tfind.XWrap(f,Pcad,fill_value=np.nan)

    ### Determine the indecies of the points to fit. ###
    ms   = np.arange( f0W.shape[0] ) * Pcad + epochcad

    # Exclude regions where the convolution returned a nan.
    sLDT = [slice(m - wdcad/2 , m+wdcad/2) for m in ms]

    return sLDT


def hist(tres):
    """
    Histogram
    
    Shows the S/N ratio of the highest significance peak in the periodogram.
    """
    assert len(unique(tres.Pblock))==1,'Periods must be the same'
    assert len(unique(tres.KIC))==1,'Must compare the same star'

    KIC = unique(tres.KIC)[0]
    dfL = unique(tres.df)

    fig,axL = subplots(nrows=len(dfL),sharex=True,figsize=( 5,  12))
    for df,ax in zip(dfL,axL):
        tg = tres.where( (tres.df == df) & tres.bg)
        tb = tres.where( (tres.df == df) & ~tres.bg)
        ax.hist(tg.os2n,color='green',bins=arange(100),
                label='%d' % len(tg.data))
        ax.hist(tb.os2n,color='red',bins=arange(100),
                label='%d' % len(tb.data))
        ax.legend()

        tprop = dict(size=10,name='monospace')
        at = AnchoredText(r"%i ppm" % df,prop=tprop, frameon=True,loc=3)
        ax.add_artist(at)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper'))

    xlabel('s2n')

def simplots(tres):
    PL = unique(tres.Pblock)
    fcount = 0 
    for P in PL:
        t = tres.where(tres.Pblock == P)
        hist(t)
        fig = gcf()
        fig.savefig('%02d.png' % fcount )
        fcount +=1
        fig.clf()

    for P in PL:
        t = tres.where(tres.Pblock == P)
        ROC(t)
        fig = gcf()
        fig.savefig('%02d.png' % fcount )
        fcount +=1
        fig.clf()

def inspSim():

    tLC = atpy.Table('tLC.fits')
    tRED = atpy.Table('tRED.fits')
    P   = np.unique(tRED.Pblock)
    dfL = np.unique(tRED.df)
    tfail = tRED.where((tRED.df == dfL[1]) & (tRED.Pblock == P[2]) &  ~tRED.bg)
    LDTfail = []
    seeds = []
    nl    = []
    Aseeds = tfail.seed
    for seed in Aseeds:
        try:
            tPAR = tRED.where(tRED.seed == seed)
            tRES = atpy.Table('tRES%04d.fits' % seed)

            ikwn = argmin(abs( tRES.PG - tPAR.P  ))
            nT = tRES.nT[0][ikwn]

            nl.append(nT)        
            sketch.inspFail(tPAR,tLC,tRES)
            fig = gcf()
            fig.savefig('insp%04d.png' % tPAR.seed)
            close('all')
        except ValueError:
            LDTfail.append( tPAR.seed[0] )


def inspVAL(tLC,tRES,*pL):
    f = tLC.fdt - tLC.fcbv
    t = tLC.t

    nrows = 4 + 2*len(pL)

    fig = gcf()
    fig.clf()
    ax0 = fig.add_subplot(nrows,1,1)
    ax1 = fig.add_subplot(nrows,1,2,sharex = ax0)
    ax0.plot(t,f)

    fm = ma.masked_invalid(f)
    fm.fill_value=0

    dM = tfind.mtd(t,fm.filled(),14)
    dM.fill_value = np.nan
    dM.mask = fm.mask | ~tfind.isfilled(t,f,14)
    
    ax1.plot(t,dM)

    ax2 = fig.add_subplot(nrows,1,3)
    ax3 = fig.add_subplot(nrows,1,4,sharex = ax2)

    sca(ax2)
    periodogram(tRES)

    sca(ax3)
    pep(tRES)

    axL = [ax0,ax1,ax2,ax3]
    for i in range(5,nrows+1):
        axL.append( fig.add_subplot(nrows,1,i) )


    for i in range(len(pL)):
        p = pL[i]

        ifom = 4+2*i
        ildt = 5+2*i

        sca( axL[ifom] )

        FOM(tLC.t[0],dM,p['P'])

        epoch = p['epoch']+np.ceil((tLC.t[0]-p['epoch'])/p['P'])*p['P']
        axvline(epoch)
        sca( axL[ildt] )
        LDT(t,f,p)


    plt.subplots_adjust(hspace=0.16)
    
    draw()


def pep(tRES):
    """
    Show best epoch as a function of period
    """
    ax = gca()

    x = tRES.PG
    y = tRES.epoch
    c = tRES.s2n
    sid = argsort(c)

    x = x[sid]
    y = y[sid]
    c = c[sid]
    ax.scatter(x,y,c=c,cmap=cm.gray_r,edgecolors='none',vmin=7)

def periodogram(tRES):
    ax = gca()
    x = tRES.PG
    y = tRES.s2n
    ax.plot(x,y)


def dMLDT(t,f,p,axL):
    """
    Plot folded mean depth and local detrending
    """
    assert axL.size==2 

    sca(axL[0])
    FOM(dM,pknown['P'])
    axvline(pknown['epoch']/lc)
    sca(axL[1])
    LDT(tLC.t,f,p)
        

def pp(tLCbase,tLC):
    """
    """
    fig,axL = subplots(nrows=5,sharex=True)
    fig.subplots_adjust(hspace=0.0001,bottom=0.03,top=0.97,left=0.06,right=0.97)
    ll = [axL[0].plot(t.TIME,t.f,',r',mew=0) for t in tLCbase]
    ll[0][0].set_label('Original Time Series')
    axL[0].plot(tLC.TIME,tLC.f,',k',mew=0,label='Pre-processing')

    axL[1].plot(tLC.TIME,tLC.fdtm,',k',mew=0,label='Filtered Data')
    axL[1].plot(tLC.TIME,tLC.fcbv,'r',mew=0,label='CBV detrend')


    dM,x,x,x,x = tfind.MF(tLC.f,20)
    dMcbv,x,x,x,x = tfind.MF(tLC.f-tLC.fcbv,20)
    axL[2].plot(tLC.TIME,dM,'k')
    axL[2].plot(tLC.TIME,dMcbv,'r')

    sca(axL[3])
    waterfall(tLC.TIME,tLC.f,cmap=cm.hot )
    ylabel('Specgram LC')
    sca(axL[4])
    waterfall(tLC.TIME,tLC.fcbv,cmap=cm.hot)
    ylabel('Specgram DT')
    
def waterfall(t,f,**kwargs):
    fm = ma.masked_invalid(f)
    fmdt = fm.copy()
    fmdt.fill_value=0
    sL = ma.notmasked_contiguous(fm)

    dt3 = lambda x,y: y - polyval(polyfit(x,y,3),x)
    for s in sL:
        fmdt[s] = dt3(t[s],fm[s])
    fdt0 = fmdt.filled()
    
    n = 10
    NFFT = 2**n
    Fs = 48

    Pxx, freqs, bins, im = \
        specgram(fdt0, NFFT=2**n, Fs=Fs,xextent=(t[0],t[-1]) ,
                 interpolation='nearest',scale_by_freq=False,pad_to=2**14,
                 noverlap=2**n-2**6)    
    cla()

    fMaId = argsort(abs(freqs-1))[0]
    fMa = freqs[fMaId]    
    Pxx = Pxx[:fMaId,::]
    per = percentile(Pxx,50)
    bins += t[0] - NFFT/ 2 /Fs

    imshow(Pxx,aspect='auto',extent=[bins[0],bins[-1],0,freqs[fMaId]],
           origin='left',vmin=per,**kwargs)

    
def phfold(t,fm,p,**kwargs):
    """

    """
    p1L,idL =  tval.LDT(t,fm,p)
    for p1,id in zip(p1L,idL):
        trend = keptoy.trend(p1[3:],t[id])
        ffit  = keptoy.P051T(p1,t[id])
        tmod = mod(t[id]-t[0],p['P'])+t[0]
        scatter(tmod,fm[id]-trend,**kwargs)
    plot(tmod,ffit-trend,'--',lw=2)


def flux(tLC,step=100,type='flux'):
    start = np.floor(tLC.t[0] / step) * step
    stop  = np.ceil(tLC.t[-1] / step) * step
    nstep = stop  / step 

    fdt = ma.masked_array(tLC.fdt,tLC.fmask)
    fcbv = ma.masked_array(tLC.fcbv,tLC.fmask)

    fm = fdt - fcbv
    dM = tfind.mtd(tLC.t,tLC.fdt-tLC.fcbv,tLC.isStep,tLC.fmask,20 )

    lineL = ['isBadReg','isStep','isDis']
    d = {}
    for line in lineL:
        x = tLC[line]
        d[line] = ma.masked_array(zeros(x.size),~x)

    vstep = ma.median(ma.abs(fdt))*10
    rcParams['axes.prop_cycle'] = ['c','m','green']

    if type is 'flux':
        for i in np.arange(nstep):
            plot(tLC.t-i*step,fdt  - vstep*i,',k')
            plot(tLC.t-i*step,fcbv - vstep*i,'r')
            for line in lineL:
                if i ==(nstep - 1):
                    label=line
                else: 
                    label=None
        
                x = d[line]
                plot(tLC.t-i*step,x - vstep*(i-0.3),lw=4,label=label)
        legend()

    elif type is 'dt':
        for i in np.arange(nstep):
            plot(tLC.t-i*step,fm  - vstep*i,',')
            plot(tLC.t-i*step,dM - vstep*i)


    xlim(-10,110)
    axvline(0)
    axvline(step)
