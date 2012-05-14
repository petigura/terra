"""
Transit finder.

Evaluate a figure of merit at each point in P,epoch,tdur space.
"""
from scipy import ndimage as nd
import scipy
import sys
import numpy as np
from numpy import ma
from matplotlib import mlab

from keptoy import *
import keptoy
import keplerio
import detrend

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

def isfilled(t,fm,twd):
    """
    Is putative transit filled?  This means:
    1 - Transit > 25% filled
    2 - L & R wings are both > 25% filled
    """
    assert keplerio.iscadFill(t,fm.data),'Series might not be evenly sampled'

    bK,boxK,tK,aK,dK = GenK(twd ) 
    bgood = (~fm.mask).astype(float) 

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

    if type(x) is np.ma.core.MaskedArray:
        pad = ma.empty(nExtend)
        pad.mask = True
        x = ma.hstack( (x ,pad) )
    else:    
        pad = np.empty(nExtend) 
        pad[:] = fill_value
        x = np.hstack( (x ,pad) )
    xwrap = x.reshape( nrow,-1 )

    return xwrap


def P2Pcad(PG0):
    """
    Period Grid (cadences)
    """
    assert type(PG0) is np.ndarray, "Period Grid must be an array"

    PcadG = (np.round(PG0/lc)).astype(int)
    PG = PcadG * lc

    return PcadG,PG

def mtd(t,f,isStep,fmask,twd):
    """
    Mean Transit Depth

    Convolve time series with our locally detrended matched filter.  

    Parameters
    ----------
    t      : time series 
    f      : flux series.  f can contain no nans since invade good region 
             during convolution.convolution. Interpolate through them. 
    isStep : Boolean array specifying the step discontinuities in the data.  
             They will be grown by the width of the convolution kernel
    fmask  : mask specifying which points are valid.
    twd    : Width of kernel in cadances

    Notes
    -----
    Since a single nan in the convolution kernel will return a nan, we
    interpolate the entire time series.  We see some edge effects

    """
    
    # If f contains nans, interpolate through them
    if f[np.isnan(f)].size > 0:
        t,f = detrend.maskIntrp( t , ma.masked_invalid(f) )

    bK,boxK,tK,aK,dK = GenK( twd )
    nK = dK.size
    dM = nd.convolve1d(f,dK)

    # Grow step mask.
    isStep = nd.convolve( isStep.astype(int) , np.ones(nK) )
    isStep = isStep > 0

    fm   = ma.masked_array(f,mask=fmask) 
    mask = ~isfilled(t,fm,twd) | isStep
    dM = ma.masked_array(dM,mask=mask,fill_value=0)
    return dM

def tdpep(t,fm,isStep,PG0):
    """
    Transit-duration - Period - Epoch

    Parameters 
    ---------- 
    fm  : Flux with bad data points masked out.  It is assumed that
          elements of f are evenly spaced in time.
    PG0 : Initial period grid.

    Returns
    -------

    epoch2d : Grid (twd,P) of best epoch 
    df2d    : Grid (twd,P) of depth epoch 
    count2d : number of filled data for particular (twd,P)
    noise   : Grid (twd) typical scatter 
    PG      : The Period grid
    twd     : Grid of trial transit widths.

    """
    assert fm.fill_value ==0
    # Determine the grid of periods that corresponds to integer
    # multiples of cadence values
    PcadG,PG = P2Pcad(PG0)
       
    # Initialize tdur grid.  
    twdMi = a2tdur( P2a( PG[0 ] ) ) /keptoy.lc
    twdMa = a2tdur( P2a( PG[-1] ) ) /keptoy.lc
    twdG = np.round(np.linspace(twdMi,twdMa,4)).astype(int)

    rec2d = []
    noise = []
    for twd in twdG:
        dM = mtd(t,fm.filled(),isStep,fm.mask,twd)
        rec2d.append( pep(t[0],dM,PcadG) )

        # Noise per transit 
        mad = ma.abs(dM)
        mad = ma.median(mad)
        noise.append(mad)

    rec2d = np.vstack(rec2d)

    make2d = lambda x : np.tile( np.vstack(x), (1,rec2d.shape[1] ))
    rec2d = mlab.rec_append_fields(rec2d,'noise',make2d(noise))
    rec2d = mlab.rec_append_fields(rec2d,'twd',  make2d(twdG))

    PG = np.tile( PG, (rec2d.shape[0],1 ))
    rec2d = mlab.rec_append_fields(rec2d,'PG',PG)

    s2n   = rec2d['fom']/rec2d['noise']*rec2d['count']
    rec2d = mlab.rec_append_fields(rec2d,'s2n',  s2n )
    return rec2d

def pep(t0,dM,PcadG):
    """
    Period-Epoch

    Wraps ep over a grid of periods.  It marginalizes over epoch.

    Parameters
    ----------
    t0    : time of first dM[0].
    dM    : depth statistic
    PcadG : Grid of periods (units of cadance)
    """

    func = lambda Pcad: ep(t0,dM,Pcad)
    resL = map(func,PcadG)

    # Marginalize over epoch.
    func = lambda r,i : (r['epoch'][i],r['fom'][i],r['count'][i])
    iMa = [ np.argmax(r['fom']) for r in resL ]
    res = map(func,resL,iMa)
    res = array(res,dtype=[('epoch',float),('fom',float),('count',int)])

    return res

def ep(t0,dM,Pcad):
    """
    Search in Epoch.

    Parameters
    ----------
    t0   : Time of first cadance.  This is needed to set the epoch.
    dM   : Transit depth estimator
    Pcad : Number of cadances to foldon

    Returns the following information:
    - 'fom'    : Figure of merit for each trial epoch
    - 'count'  : 
    - 'epoch'  : Trial 
    - 'win'    : Which epochs passed (window function)
    """
    
    dMW = XWrap(dM,Pcad)
    nt,ne = dMW.shape
    epoch = np.arange(ne,dtype=float)/ne * Pcad *lc + t0

    # Gives the cuts that all transits must pass to be included in the MES
    vcount = (~dMW.mask).astype(int).sum(axis=0)
    sig = (dM > 50e-6).astype(int)
    sigW = XWrap(sig,Pcad,fill_value=0)
    nsig = sigW.sum(axis=0)
    bsig = (nsig == vcount).astype(float)

    # The fact that we 
    win = (vcount >= 3)  # Did we see at least 3 transits?

    cut = win & (nsig == vcount) # The cuts themselves.

    win = win.astype(float)
    cut = cut.astype(float)

    fom = cut*dMW.mean(axis=0)
    
    res = {
        'fom'    : fom         ,
        'epoch'  : epoch       ,
        'win'    : win         ,
        'count'  : vcount        ,
        }

    return res

def tdmarg(rec2d):
    """
    tdur marginalize.

    Marginalize over the transit duration.

    Parameters
    ----------
    t   : Time series
    f   : Flux
    PG0 : Initial Period Grid (actual periods are integer multiples of lc)

    Returns
    -------
    rec : Values corresponding to maximal s2n:
    
    """
    # Marginalize over tdur
    iMaTwd = np.argmax(rec2d['s2n'],axis=0)
    x      = np.arange(rec2d.shape[1])
    rec    = rec2d[iMaTwd,x]

    return rec



