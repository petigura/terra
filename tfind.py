"""
Transit finder.

Evaluate a figure of merit at each point in P,epoch,tdur space.
"""
from scipy import ndimage as nd
import scipy
import sys
import numpy as np
from numpy import ma
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

def P2Pcad(PG0):
    """
    Period Grid (cadences)

    """
    assert type(PG0) is np.ndarray, "Period Grid must be an array"

    PcadG = (np.round(PG0/lc)).astype(int)
    PG = PcadG * lc

    return PcadG,PG

def tdpep(fsig,PG0):
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

        dd,ee, cc, ss =  pep(dM,PcadG)

        eee.append(ee)
        ddd.append(dd)
        sss.append(ss)
        ccc.append(cc)

    eee = np.vstack( [np.array(ee) for ee in eee] )
    ddd = np.vstack( [np.array(dd) for dd in ddd] )
    sss = np.vstack( [np.array(ss) for ss in sss] )
    ccc = np.vstack( [np.array(cc) for cc in ccc] )

    return eee,ddd,sss,ccc,PG

def pep(dM,PcadG):
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
    epoch,df,noise,nT,PG = tdpep(f,PG0)
    iMaTwd = np.argmax(df/noise,axis=0)
    x      = np.arange(PG0.size)

    epoch = epoch[iMaTwd,x]
    df = df[iMaTwd,x]
    noise = noise[iMaTwd,x]
    nT = nT[iMaTwd,x]
    s2n = df/noise

    res = {'epoch':epoch,'df':df,'noise':noise,'nT':nT,'PG':PG,'s2n':s2n}

    return res
