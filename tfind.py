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
import keplerio

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

def isfilled(t,f,twd):
    """
    Is putative transit filled?  This means:
    1 - Transit > 25% filled
    2 - L & R wings are both > 25% filled
    """
    assert keplerio.iscadFill(t,f),'Series might not be evenly sampled'

    bK,boxK,tK,aK,dK = GenK(twd ) 
    dM,bM,aM,DfDt,f0 = MF(f,20)
    fn = ma.masked_invalid(f)
    bgood = (~fn.mask).astype(float) 

    bfl = nd.convolve(bgood,bK) # Compute the filled fraction.
    afl = nd.convolve(bgood,aK)
    tfl = nd.convolve(bgood,tK)

    # True -- there is enough data for us to look at a transit at the
    # midpoint
    filled = (bfl > 0.25) & (afl > 0.25) & (tfl > 0.25) 

    return filled

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
    tfold = time - epoch # Slide transits to 0, P, 2P
    tfold = np.mod(tfold,P)
    tfold = ma.masked_inside(tfold,wd/2,P-wd/2)
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

def tdpep(t,f,PG0):
    """
    Transit-duration - Period - Epoch

    Parameters
    ----------
    f    : Flux time series.  It is assumed that elements of f are
           evenly spaced in time.

    PG0  : Initial period grid.

    Returns
    -------

    epoch2d : Grid (twd,P) of best epoch 
    df2d    : Grid (twd,P) of depth epoch 
    count2d : number of filled data for particular (twd,P)
    noise   : Grid (twd) typical scatter 
    PG      : The Period grid
    twd     : Grid of trial transit widths.

    """

    # Determine the grid of periods that corresponds to integer
    # multiples of cadence values
    PcadG,PG = P2Pcad(PG0)
       
    # Initialize tdur grid.  
    twdMi = a2tdur( P2a( PG[0 ] ) ) /lc
    twdMa = a2tdur( P2a( PG[-1] ) ) /lc

    twdG = np.round(np.linspace(twdMi,twdMa,4)).astype(int)

    func = lambda twd: pep(t,f,twd,PcadG)
    resL = map(func,twdG)

    rec2d = np.vstack([ r[0] for r in resL ])
    noise = np.array([  r[1] for r in resL ])

    res = {'epoch2d':rec2d['epoch'],
           'df2d'   :rec2d['fom'],
           'count2d':rec2d['count'],
           'PG'     :PG,
           'twd'    :twdG,
           'noise'  :noise
           }
    return res

def pep(t,f,twd,PcadG):
    """
    Search in period then epoch:
    """
    bK,boxK,tK,aK,dK = GenK( twd )
    dM = nd.convolve1d(f,dK)

    # Noise per transit 
    mad = ma.masked_invalid(dM)
    mad = ma.abs(mad)
    mad = ma.median(mad)

    # Discard cadences that are too high.
    dM = ma.masked_outside(dM,-1e-3,1e-3)
    f = ma.masked_array(f,mask=dM.mask,fill_value = np.nan)
    f = f.filled()
    dM = nd.convolve1d(f,dK)

    filled = isfilled(t,f,twd)

    func = lambda Pcad: ep(t,dM,Pcad)
    resL = map(func,PcadG)

    # Find the maximum index for every period.
    iMa = [ np.argmax(r['fom']) for r in resL  ]

    func = lambda r,i : (r['epoch'][i],r['fom'][i],r['count'][i])
    res = map(func,resL,iMa)
    res = array(res,dtype=[('epoch',float),('fom',float),('count',int)])

    noise = mad
    return res,noise

def ep(t,dM,Pcad):
    """
    Search in Epoch.


    Returns the following information:
    - 'fom'    : Figure of merit for each trial epoch
    - 'count'  : 
    - 'epoch'  : Trial 
    - 'win'    : Which epochs passed (window function)
    """
    
    dMW = XWrap(dM,Pcad,fill_value=np.nan)
    dMW = ma.masked_invalid(dMW)

    nt,ne = dMW.shape
    epoch = np.arange(ne,dtype=float)/ne * Pcad *lc + t[0]

    vcount = (~dMW.mask).astype(int).sum(axis=0)
    win = (vcount >= 3).astype(float)

    sig = (dM > 50e-6).astype(int)
    sigW = XWrap(sig,Pcad,fill_value=0)
    nsig = sigW.sum(axis=0)
    bsig = (nsig == vcount).astype(float)


    d = bsig*win*dMW.mean(axis=0)
    
    fom = d

    res = {
        'fom'    : fom         ,
        'epoch'  : epoch       ,
        'win'    : win         ,
        'count'  : vcount        ,
        }

    return res

def tfindpro(t,f,PG0):
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
    res = tdpep(t,f,PG0)

    epoch = res['epoch2d']
    df    = res['df2d']
    nT    = res['count2d']
    PG    = res['PG']
    twd   = res['twd']

    noise2d = res['noise'].reshape(twd.size,1).repeat(df.shape[1],axis=1)
    s2n = df/noise2d*sqrt(nT)

    iMaTwd = np.argmax(s2n,axis=0)
    x      = np.arange(PG0.size)

    epoch = epoch[iMaTwd,x]
    df    = df[iMaTwd,x]
    noise = noise2d[iMaTwd,x]
    nT    = nT[iMaTwd,x]
    s2n   = s2n[iMaTwd,x]
    twd   = np.array( [twd[i] for i in iMaTwd] )

    res = {'epoch':epoch,'df':df,'noise':noise,'nT':nT,'PG':PG,'s2n':s2n,
           'twd':twd}

    return res

