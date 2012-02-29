"""
A module for visualizing Kepler data.

The idea is to keep it rough.
"""
import atpy
from numpy import *
import glob
import matplotlib.pylab as plt
from matplotlib.pylab import *

from matplotlib import rcParams
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from keptoy import *
import keptoy
import qalg

def cdpp():
    t = atpy.Table('sqlite','all.db',table='ch1cdpp')

    cdppmin = min(t.cdpp12hr)
    cdppmax = max(t.cdpp12hr)

    nplots = 8
    cdpp = logspace(log10(20),log10(200),nplots)

    for i in range(nplots):
        print cdpp[i]
        closest  = argsort( abs( t.cdpp12hr - cdpp[i] ))[0]
        starcdpp = t.cdpp12hr[closest]
        keplerid = t.KEPLERID[closest]
        file = glob.glob('archive/data3/privkep/EX/Q3/kplr%09i-*_llc.fits' %
                         keplerid)

        star = atpy.Table(file[0],type='fits') 
        ax = plt.subplot(nplots,1,i+1)

        med = median(star.SAP_FLUX )

        ax.plot(star.TIME,(star.SAP_FLUX/med-1),'.',ms=2,
                label='KIC-%i, CDPP-12hr %.2f' % (keplerid,starcdpp) )
        ax.legend()



def markT(ax,tT,**kwargs):
    for t in tT:
        ax.axvline(t,**kwargs)
    return ax

    
def inspectT(t0,f0,P,ph,darr=None):
    """
    Take a quick look at a transit
    """
    size = 150
    pad = 0.1 # amount to pad in units of size
    cW = 2 
    linscale = 3
    
    fig = plt.gcf()

    f = f0.copy()
    t = t0.copy()

    f -= f.mean()
    t -= t[0]
    tbase = t.ptp()

    nt = int(ntrans( tbase, P, ph ))
    otT =  P * (arange(nt) + ph ) 
    print otT,t0[0]

    # Plot the time series
    nstack = int(ceil( t.ptp() / size))
    gs = GridSpec(2,1)

    gsStack = GridSpec(nstack, 1)
    gsStack.update(hspace=0.001,bottom=.3,left=0.03,right=0.98)
    gsT = GridSpec(1, nt)
    gsT.update(top=0.28,wspace=0.001,left=0.03,right=0.98)

    axStackl = []
    for i in range(nstack):
        axStackl.append( plt.subplot( gsStack[i]) ) 
        ax = axStackl[i]
        offset = size*i
        ax.plot(t,f,marker='.',ms=2,lw=0,alpha=.6)
        ax.set_xlim(offset-pad*size,offset+(1+pad)*size)
        ax.axvline(offset,ls='--',lw=1,label='Padded')
        ax.axvline(offset+size,ls='--',lw=1)
        ax.annotate('Offset = %i' % offset,xy=(.01,.1),
                    xycoords='axes fraction')

        xa = ax.get_xaxis()
        ya = ax.get_yaxis()

        rms = std(f)
        linthreshy = linscale*rms
        ax.set_yscale('symlog',linthreshy=linthreshy)
        ax.axhline(linthreshy,color='k',ls=':')
        ax.axhline(-linthreshy,color='k',ls=':')
        ax = markT(ax,otT,color='red',lw=3,alpha=0.4)

        if darr != None:
            inT = int(ntrans( tbase, darr['P'], darr['phase'] ))
            itT =  darr['P']*arange(inT) + darr['phase'] * darr['P'] - t0[0]
            ax = markT(ax,itT,color='green',lw=3,alpha=0.4)

        if i == 0:
            xa.set_ticks_position('top')
            ax.legend(loc='upper left')
        else:
            xa.set_visible(False)
            ya.set_visible(False)

    tdur = a2tdur(P2a(P))

    axTl = []    
    for i in range(nt):
        axTl.append( plt.subplot( gsT[i] ) )
        axT = axTl[i]
        axT.plot(t,f,'.')        

        tfit,yfit = lightcurve(tbase=tbase,phase=ph,P=P,df=f)
        axT.plot(tfit,yfit-1,color='red')
        axT.set_xlim( otT[i] - cW*tdur , otT[i] + cW*tdur )
        lims = axT.axis()

        tm = ma.masked_outside( t,lims[0],lims[1] )
        fm = ma.masked_array(f,mask=tm.mask)        

        axT.axis( ymax=fm.max() , ymin=fm.min() )        
        xticklabels = axT.get_xticklabels()
        [xtl.set_rotation(30) for xtl in xticklabels]
        ya = axT.get_yaxis()

        if i != 0:
            plt.setp(ya,visible=False)

    limarr = array([ ax.axis() for ax in axTl ])
    yMi = min(limarr[:,2])
    yMa = max(limarr[:,3])


    for ax in axTl:
        ax.axis(ymax=yMa , ymin=yMi) 
        


def stack(axL,xmin,size,pad=0.1,lfsize='small'):
    """
    Given a list of axis, we'll adjust the x limits so we can fit a very long
    data string on the computer screen.
    """
    nAx = len(axL)
    for i in range(nAx):

        ax = axL[i]
        offset = xmin + i*size

        ax.set_xlim(offset-pad*size,offset+(1+pad)*size)
        ax.axvline(offset,ls='--',label='Padded')
        ax.axvline(offset+size,ls='--')
        ax.annotate(r'$\Delta$ T = %i' % (i*size) ,xy=(.01,.1),
                    xycoords='axes fraction',fontsize=lfsize)
        
        xa = ax.get_xaxis()
        ya = ax.get_yaxis()

        if i == 0:
            xa.set_ticks_position('top')
        elif i ==nAx-1:
            ax.legend(loc='lower right')
            xa.set_visible(False)
            ya.set_visible(False)
        else:
            xa.set_visible(False)
            ya.set_visible(False)

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

from keptoy import lc
import tfind

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

def XWrap(XW,step=1):
    """
    Plot XWrap arrays, folded on the right period.
    """

    nT = XW.shape[0]
    ncad = XW.shape[1]
    [plt.plot(XW[i,:]+i*step,aa=False) for i in range(nT)]    

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

import tval
import copy

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

        color = rcParams['axes.color_cycle'][mod(i,4)]

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

def ROC(tres):
    """
    
    """
    assert len(unique(tres.Pblock))==1,'Periods must be the same'
    assert len(unique(tres.KIC))==1,'Must compare the same star'

    KIC = unique(tres.KIC)[0]
    dfL = unique(tres.df)
    
    for df in dfL:
        t = tres.where( tres.df == df)
        fapL,etaL = qalg.ROC(t)
        plot(fapL,etaL,lw=2,label='df  = %03d ' % (df) )
    
    x = linspace(0,1,100)
    plot(x,x)

    legend(loc='best')
    title( 'ROC for %i' % KIC ) 
    xlabel('FAP' )
    ylabel('Detection Efficiency' )

def hist(tres):
    """
    
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
                label='Good %d' % len(tg.data))
        ax.hist(tb.os2n,color='red',bins=arange(100),
                label='Fail %d' % len(tb.data))
        ax.legend()

        label = r"""
$\Delta F / F$  = %(df)i ppm
""" % {'df':df}

        ax.annotate(label,xy=(.8,.1),xycoords='axes fraction',
                    bbox=dict(boxstyle="round", fc="w", ec="k"))


    xlabel('s2n')
    title('%d, %i days' % (KIC,tres.Pblock[0])  )


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
