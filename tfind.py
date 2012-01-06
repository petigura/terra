"""
Transit finder: A new approach to finding low signal to noise transits.
"""
from scipy import ndimage as nd
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
import scipy
import sys
import numpy as np
from numpy import ma,nan
from numpy.polynomial import Legendre
from scipy import optimize

import ebls
from keptoy import *
from keptoy import lc 

def GenK(twdcad,fcwd=1):
    """
    Generate Kernels.  
    
    Parameters
    ----------
    twidcad : length of the transit (cadences)
    fcwid   : Each side of the continuum is fcwd * twid (default is 1)

    Returns
    -------
    bK : Before transit kernel.
    tK : Trasit kernel.
    aK : After transit kernel.
    dK : Depth kernel.

    Notes
    -----
    Recall that conv(f,g) will flip g.

    """
    cwd = twdcad * fcwd
    kSize = 2*cwd + twdcad  # Size of the kernel
    kTemp = np.zeros(kSize) # Template kernal of appropriate length.

    bK = kTemp.copy()
    bK[-cwd:] = np.ones(1) / cwd # Before transit kernel

    boxK = kTemp.copy()
    boxK[cwd:-cwd] = np.ones(1) # Box shaped kernel

    tK = boxK.copy()
    tK[cwd:-cwd] /= twdcad # transit kernel

    aK = kTemp.copy()
    aK[:cwd] = np.ones(1) / cwd # After transit kernel

    dK = kTemp.copy() 
    dK = 0.5*(bK+aK) - tK # Depth kernel

    return bK,boxK,tK,aK,dK


def MF(fsig,twd,fcwd=1):
    """
    Matched filter.

    """
    cwd = fcwd*twd

    bK,boxK,tK,aK,dK = GenK(twd,fcwd=fcwd)

    dM     = nd.convolve1d(fsig,dK)
    bM     = nd.convolve1d(fsig,bK)
    aM     = nd.convolve1d(fsig,aK)
    DfDt   = (aM-bM)/(cwd+twd)/lc
    f0     = 0.5 * (aM + bM)    # Continuum value of fsig (mid transit)

    return dM,bM,aM,DfDt,f0

def dThresh(dM,dMi=50,dMa=1000):
    """
    Depth threshold.

    Applies a cut to the possible transits based on their depth.

    Parameters
    ----------
    dM  : Mean depth of transit.
    dMa : The maximum depth (ppm).
    dMi : The minimum depth (ppm).

    Returns
    -------

    tCand : Boolean array with showing which cadances are cadidate
            mid-transit.

    """
    
    dMi *= 1e-6
    dMa *= 1e-6

    tCand = (dM > dMi) & (dM < dMa)

    return tCand

def pCheck(time,tCand,P0,tbin,nTMi=3):
    """
    Periodicity check.

    Throw out transit cadidates that are not consistent with a single
    period.

    Parameters
    ----------
    time  : Times of each cadance.
    tCand : Boolean array with showing which cadances are cadidate
            mid-transit.
    P0    : Trial period (units of days)
    tbin    : Number of phase bins to check.
    nTMi  : Require that signals repeat a certain number of times.  

    Returns
    -------
    eCand : List of epochs that show 3 transits.

    """
    assert type(tbin) == int

    tbase = np.nanmax(time)-np.nanmin(time)
    epoch = np.arange( tbin,dtype=float ) / tbin * P0
    nT = np.ceil(tbase/P0)
    timeInt = np.arange(tbin*nT)/tbin*P0
    tCandInt = interp1d(time,tCand,kind='nearest',bounds_error=False,
                        fill_value=0)
    
    tCandRS = tCandInt(timeInt).reshape(nT,tbin)
    boolT = tCandRS.sum(axis=0) >= nTMi
    eCand = epoch[np.where(boolT)]

    return eCand

def XWrap(x,ifold,fill_value=0):
    """
    Extend and wrap array.
    
    Fold array every y indecies.  There will typically be a hanging
    part of the array.  This is padded out.

    Parameters
    ----------

    x     : input
    ifold : Wrap array after ifold indecies.

    Return
    ------

    xwrap : Wrapped array.

    """

    ncad = x.size # Number of cadences
    nrow = int(np.floor(ncad/ifold) + 1)
    nExtend = nrow * ifold - ncad # Pad out remainder of array with 0s.
 
    pad = np.empty(nExtend) 
    pad[:] = fill_value
    x = np.append( x ,pad )
    xwrap = x.reshape( nrow,-1 )

    return xwrap

def lseg(x,P,ph,wid):
    """
    List of segments.

    Fold input according to P.  Return segments of the array at a
    specific phase of a specific width.

    Parameters
    ----------
    x   : Array to be extracted
    P   : Fold array every P indecies (cad)
    ph  : Return every points at ph. (cad)
    wid : Width of each returned segment (cad)

    """
    
    sl = 0 # low end of the slice.
    sh = 0 # Upper end of the slice.
    iseg = 0
    xlseg = []

    n = len(x)
    while sl < n:
        sl = P * iseg + ph - wid/2
        sh = P * iseg + ph + wid/2
        xlseg.append(x[sl:sh])
        iseg += 1

    return xlseg




def boxBR(fsig,dTrans,sig,boxK):
    """
    Box Bayes Ratio.

    Compute the Bayes ratio of Prob( transit ) / Prob( flat ).


    Parameters
    ----------
    fsig   : Detrended flux time series.
    dTrans : Depth of transit.
    boxK   : Model constructed from the same kernel.
    sig    : Point by point sigma.

    Returns
    -------
    BR : log of Bayes Ratio.

    """

    # Compute Chi^2 for the transit model
    tMod = -1.*depth*boxK
    tChi2 = Chi2(fsig,tMod,sig)

    # Compute Chi^2 for the null model
    nlMod = 0.*boxK
    nlChi2 = Chi2(fsig,nlMod,sig)

    BR = tChi2 - nlChi2 

    return BR


def Chi2(data,model,sig):
    """
    Chi^2 with constant sigma.
    """
    
    r = data - model
    nr = r/sig
    chi2 = (nr**2).sum()

    return chi2

def taylorDT(fs,ts,t0,DfDt,f0):
    """
    Taylor series detrending.
    
    Perform a local detrending at a particular cadence.

    Arguments
    ---------
    fs   : Flux segment.
    ts   : Time segment.
    t0   : Time we're expanding about.
    DfDt : Local slope of lightcurve (determined from continuum).
    f0   : Local flux level of continuum.

    Returns
    -------
    trend : Fluxes corresponding to the detrended light curve.

    """
    nptsMi = 10.
    assert fs.size == ts.size, "ts and fs must have equal sizes"
    assert fs.size > nptsMi, "Not enough points"

    trend =  DfDt*(ts - t0) + f0

    return trend


def seg(time,P,ph,wid):
    """
    Segment 

    Return a list of slices corresponding to P,ph and have width = wid

    Arguments
    ---------
    time : numpy array (not masked).
    P    : Period to fold about (days)
    ph   : phase (offset from t = 0)/P
    wid  : Width (days)

    Returns
    -------
    List of slices corresponding to the segment.

    Notes 
    ----- 
    Will not return segments that are shorter than the specified
    width.  If there is missing data, interpolate over it or lose it.
    
    """
    tph = ph*P # time phase.

    tfold = np.mod(time,P)
    tfold = ma.masked_invalid(tfold)
    tfold = ma.masked_outside(tfold,tph-wid/2.,tph+wid/2.)
    sL = ma.notmasked_contiguous(tfold)
    ssL = []
    for s in sL:
        if time[s].ptp() >= wid/2.:
            ssL.append(s)

    return ssL

def LDT(time,fsig,DfDt,f0,wd):
    """
    Local Detrending

    Detrend according to the local values of the continuum

    Parameters
    ----------
    time : Folded time series with all but the tranist region masked out.
    fsig : Flux time series.
    
    Returns
    -------
    fdt   : Detrended flux segments
    trend : The trend I subtracted from them.
    """
    assert type(time) is ma.core.MaskedArray, \
        "time must be masked array of transit segments"

    # Detrend the lightcurve.
    sLDT = ma.notmasked_contiguous(time)
    sLDT = [ s for s in sLDT if s.stop-s.start > wd / lc / 2 ]

    trend = ma.masked_array(fsig,copy=True,mask=True)

    for s in sLDT:
        ms = s.start + wd /lc/2
        trend[s] = taylorDT(fsig[s],time[s],time[ms],DfDt[ms],f0[ms])

    return trend

def getT(time,P,epoch,wd):
    """
    Get Transits

    time : Time series
    P    : Period
    epoch: epoch
    wd   : How much data to return for each slice.

    Returns
    -------
    Time series phase folded with everything but the transits masked out.

    """
    tfold = np.mod(time,P)
    tfold = ma.masked_outside(tfold,epoch - wd/2,epoch + wd/2)
    tfold = ma.masked_invalid(tfold)
    return tfold

def FOM(time,fsig,DfDt,f0,P,epoch,wd=2,twd=.3,plot=False):
    tfold = getT(time,P,epoch,wd)
    time = ma.masked_array(time,mask=tfold.mask)

    sLDT = ma.notmasked_contiguous(time)
    sLDT = [ s for s in sLDT if s.stop-s.start > wd / lc / 2 ]
    fdt,trend = LDT(time,fsig,DfDt,f0,wd)

    # Calculate figure of merit.
    tfold = getT(time,P,epoch,twd)
    time = ma.masked_array(time,mask=tfold.mask)
    
    fdt.mask = True
    fdt.mask = tfold.mask
    fdt = ma.masked_invalid(fdt)

    sLFOM = ma.notmasked_contiguous(fdt)
         
    if fdt.count() > 10:
        s2n = -ma.mean(fdt)/ma.std(fdt)*np.sqrt(fdt.count())
    else:
        s2n = 0

    if plot:
        fig = plt.gcf()
        fig.clf()
        ax = plt.gca()

        nT = len(sLDT)


        x = tfold.data
        ofst = 1e-3*np.arange(1,nT+1)

        for i in range(nT):
            o = ofst[i]
            sDT = sLDT[i]
            sFOM = sLFOM[i]
            mT = np.mean(trend[sDT])


            ax.plot(x[ sDT  ] ,fsig[sDT]  + o -mT,',k') 
            ax.plot(x[ sFOM ] ,fsig[sFOM] + o -mT,'or',ms=2)
            ax.plot(x[ sDT  ] ,trend[sDT] + o -mT) 

            ax.axvline(x[s.start + wd/lc/2])

            ax.plot(x[sDT],fdt.data[sDT],'o',alpha=.3)
            ax.plot(x[sDT],fdt.data[sDT],'o') 

        ax.set_title("epoch = %f" % epoch)
        ax.annotate("s2n = %f" % s2n,(.8,.8),
                    xycoords='figure fraction',
                    bbox=dict(boxstyle="round", fc="0.8"))

        fig.savefig("test.png" )

    return s2n


def pep(time,fsig,twd,cwd):
    """
    Period-epoch search
    
    Parameters
    ----------
    time - time series
    fsig - flux
    twd - transit width (in days)
    cwd - continuum width (each side, days)
    """

    twd = int(twd/lc) # length of transit in units of cadence
    bK,boxK,tK,aK,dK = GenK( twd ) 
    dM , bM   , aM   , DfDt , f0 = MF(fsig,twd)

    # Discard cadences that are too high.
    dM = ma.masked_outside(dM,-1e-3,1e-3)
    fsig = ma.masked_array(fsig,mask=dM.mask,fill_value = nan)
    fsig = fsig.filled()

    tCand = dThresh(dM,dMi=100)
    
    # Generate a grid of plausable periods (days)
    PGrid = ebls.grid( np.nanmax(time) - np.nanmin(time) , 0.5, Pmin=50, 
                       Psmp=0.5 )

    eCandGrid = []
    PCandGrid = []

    for P in PGrid:
        eCand = pCheck(time,tCand,P,1000)
        eCandGrid.append( eCand )
        PCand = np.empty( len(eCand) )
        PCand[:] = P
        PCandGrid.append( PCand )


    ee = reduce(np.append,eCandGrid)
    PP = reduce(np.append,PCandGrid)
    print "%i (P,epoch)" % len(ee)

    s2n = np.zeros(len(ee))
    for i in range(len(ee)):
        ep = ee[i]
        P  = PP[i]
        s2n[i] = FOM(time,fsig,DfDt,f0,P,ep,twd=.3,wid=.9)

    return ee,PP,s2n

def P2Pcad(PG0):
    """
    Period Grid (cadences)

    """
    assert type(PG0) is np.ndarray, "Period Grid must be an array"

    PcadG = (np.round(PG0/lc)).astype(int)
    PG = PcadG * lc

    return PcadG,PG


def tdpep2(fsig,PG0):
    """
    Transit-duration - Period - Epoch

    Parameters
    ----------
    fsig - Flux time series.  It is assumed that elements of fsig are
           evenly spaced in time.

    PG0  - Initial period grid.

    Returns
    -------

    eee - epoch of maximum depth for a paticular (twd,P)
    ddd - depth of maximum depth for a paticular (twd,P)
    sss - typical scatter for a paticular (twd,P)
    ccc - number of filled data for particular (twd,P)
    """

    # Determine the grid of periods that corresponds to integer
    # multiples of cadence values
    PcadG,PG = P2Pcad(PG0)
       
    # Initialize tdur grid.  
    twdMi = a2tdur( P2a( PG[0 ] ) ) /lc
    twdMa = a2tdur( P2a( PG[-1] ) ) /lc

    twdG = np.round(np.linspace(twdMi,twdMa,4)).astype(int)
    print twdG


    eee = []
    ddd = []
    sss = []
    ccc = []

    for twd in twdG:
        bK,boxK,tK,aK,dK = GenK( twd )
        dM = nd.convolve1d(fsig,dK)

        # Discard cadences that are too high.
        dM = ma.masked_outside(dM,-1e-3,1e-3)
        fsig = ma.masked_array(fsig,mask=dM.mask,fill_value = np.nan)
        fsig = fsig.filled()
        dM = nd.convolve1d(fsig,dK)

        dd,ee, cc, ss =  pep2(dM,PcadG)

        eee.append(ee)
        ddd.append(dd)
        sss.append(ss)
        ccc.append(cc)

    eee = np.vstack( [np.array(ee) for ee in eee] )
    ddd = np.vstack( [np.array(dd) for dd in ddd] )
    sss = np.vstack( [np.array(ss) for ss in sss] )
    ccc = np.vstack( [np.array(cc) for cc in ccc] )

    return eee,ddd,sss,ccc,PG


def pep2(dM,PcadG):
    """
    Period-epoch search
    
    Parameters
    ----------
    dM    - mean depth
    PcadG - Grid of trial periods (in units of cadences). 

    Returns
    -------
    dd    - Average depth of folded transit.
    ee    - Epoch of maximum signal strength
    cc    - Number of transits in peak signal.
    ss    - Robust scatter for a particular peroid.

    """

    ee = []
    dd = []
    ss = []
    cc = []
    for Pcad in PcadG:

        dMW = XWrap(dM,Pcad,fill_value=np.nan)
        dMW = ma.masked_invalid(dMW)
        nt,ne = dMW.shape

        epoch = np.arange(ne,dtype=float)/ne * Pcad *lc 
        count = (~dMW.mask).astype(int).sum(axis=0)
        bcount = count >= 3

        d = bcount*dMW.mean(axis=0)

        mad = ma.abs(d)
        mad = ma.masked_less(d,1e-6)
        mad = ma.median(mad)

        iMax = d.argmax()

        ee.append( epoch[iMax] )
        dd.append( d[iMax]     )
        ss.append( mad         )
        cc.append( count[iMax] )

    return  dd,ee, cc, ss

def tdpep(time,fsig):
    """
    Transit-duration - Period - Epoch
    """

    tdur = np.linspace(0.3,0.8,4)
    tt = []
    PP = []
    ee = []
    s2n = []

    for t in tdur:
        e,P,s = pep(time,fsig,t,t)
        PP.append(P)
        ee.append(e)
        s2n.append(s)
        tt.append(np.zeros(len(P))+t)
        
    tt = reduce(np.append,tt)
    PP = reduce(np.append,PP)
    ee = reduce(np.append,ee)
    s2n = reduce(np.append,s2n)

    return tt,PP,ee,s2n




def cadFill(cad0):
    """
    Cadence Fill

    We want the elements of the arrays to be evenly sampled so that
    phase folding is equivalent to array reshaping.

    Parameters
    ----------
    cad : Array of cadence identifiers.
    
    Returns
    -------
    cad   : New array of cadences (without gaps).
    iFill : Indecies that were not missing.

    """
    
    bins = np.arange(cad0[0],cad0[-1]+2)
    count,cad = np.histogram(cad0,bins=bins)
    iFill = np.where(count == 1)[0]
    
    return cad,iFill

def tfindpro(t,f,PG0,i):
    """
    Transit Finder Profiler

    Parameters
    ----------
    t   : Time series
    f   : Flux
    PG0 : Initial Period Grid (actual periods are integer multiples of lc)
    i   : The trial number
    
    Returns
    -------
    res : Dictionary of results for subsequent interpretation.
    
    """
    sys.stderr.write("%i\n" % i)
    epoch,df,noise,nT,PG = tdpep2(f,PG0)
    iMaTwd = np.argmax(df/noise,axis=0)
    x      = np.arange(PG0.size)

    epoch = epoch[iMaTwd,x]
    df = df[iMaTwd,x]
    noise = noise[iMaTwd,x]
    nT = nT[iMaTwd,x]
    s2n = df/noise

    res = {'epoch':epoch,'df':df,'noise':noise,'nT':nT,'PG':PG,'s2n':s2n}

    return res


def NLDT(t,f,epoch,f0,DfDt):
    """
    Non-linear detrend.

    Fit a single transit as a combination of a conintuum described by
    a Legendre polynomial and a box-shaped transit.

    Parameters
    ----------
    t    : time array (only the segment to be fit)
    f    : flux array (same)
    f0   : Guess for constant term
    DfDt : Guess for slope
    
    Returns
    -------
    
    """
    assert ( type(t) is np.ndarray ) & ( type(f) is np.ndarray ) \
        , "Time must be array"

    # Guess for model parameters.  3rd order Legendre poly
    p0 = [epoch, 0., 0.5, f0, DfDt ,0,0 ]

    p1, fopt ,iter ,funcalls, warnflag = \
        optimize.fmin(err,p0,args=(t,f),maxiter=1000,maxfun=1000,full_output=True)

    assert warnflag == 0, "Optimization failed "

    return p1


def tmodel(p,t):
    """
    Transit Model

    Parameters
    ----------
    p : parameters.  [ epoch, df, tdur, Legendre coeff ... ] 
    t : time

    Returns
    -------
    fmod : flux model
    
    """
    P     = p[0]
    epoch = p[1]
    df    = p[2]
    tdur  = p[3]

    domain = [t.min(),t.max()]
#    import pdb;pdb.set_trace()

    # The rest of the parameters go the continuum fit
    cont = Legendre( p[4:],domain=domain )(t)
    fmod = inject(t,cont,P=P,epoch=epoch,df=df,tdur=tdur)

    return fmod

def err(p,t,f):
    fmod = tmodel(p,t)
    return ((fmod - f)**2).sum()
#    return np.median(np.abs(fmod-f))



def NLDTWrap(time,fsig,DfDt,f0,wd):
    """
    Local Detrending

    Detrend according to the local values of the continuum

    Parameters
    ----------
    time : Folded time series with all but the tranist region masked out.
    fsig : Flux time series.
    DfDt : Derivative array (Guess value for the slope)
    f0   : f0 (guess for conintuum level)

    Returns
    -------
    fdt   : Detrended flux segments
    trend : The trend I subtracted from them.
    """

    assert type( ma.core.MaskedArray ) , "time should be masked array"

    sLDT = ma.notmasked_contiguous(time)
    sLDT = [ s for s in sLDT if s.stop-s.start > wd / lc / 2 ]

    trend = ma.masked_array(fsig,copy=True,mask=True)

    for s in sLDT:
        ms = s.start + wd /lc/2
        p1 = NLDT(time.data[s],fsig[s],time.data[ms],f0[ms],DfDt[ms])
        trend[s] = Legendre(p1[3:],domain=domain)(time[s])

    return trend
