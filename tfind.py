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
import FFA_cy as FFA

from config import *

# dtype of the record array returned from ep()
epnames = ['mean','count','epoch','Pcad']
epdtype = zip(epnames,[float]*len(epnames) )
epdtype = np.dtype(epdtype)

# dtype of the record array returned from tdpep()
tdnames = epnames + ['noise','s2n','twd']
tddtype = zip(tdnames,[float]*len(tdnames))
tddtype = np.dtype(tddtype)

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




def perGrid(tbase,ftdurmi,Pmin=100.,Pmax=None):
    """
    Period Grid

    Create a grid of trial periods (days).

    [P_0, P_1, ... P_N]
    
    Suppose there is a tranit at P_T.  We want our grid to be sampled
    such that we wont skip over it.  Suppose the closest grid point is
    off by dP.  When we fold on P_T + dP, the transits will be spread
    out over dT = N_T * dP where N_T are the number of transits in the
    timeseries (~ tbase / P_T).  We chose period spacing so that dT is
    a small fraction of expected transit duration.

    Parameters
    ----------
    tbase    : Length of the timeseries
    ftdurmi  : Minimum fraction of tdur that we'll look for. The last
               transit of neighboring periods in grid must only be off
               by a `ftdurmi` fraction of a transit duration.
    Pmax     : Maximum period.  Defaults to tbase/2, the maximum period
               that has a possibility of having 3 transits
    Pmin     : Minumum period in grid 

    Returns
    -------
    PG       : Period grid.
    """

    if Pmax == None:
        Pmax = tbase/2.

    P0  = Pmin
    PG  = []
    while P0 < Pmax:
        # Expected transit duration for P0.
        tdur   = a2tdur( P2a(P0)  ) 
        tdurmi = ftdurmi * tdur
        dP     = tdurmi / tbase * P0
        P0 += dP
        PG.append(P0)

    PG = np.array(PG)
    return PG

def P2Pcad(PG0,ncad):
    """
    Convert units of period grid from days to cadences

    We compute MES by averaging SES column-wise across a wrapped SES
    array.  We must fold according to an integer number of cadences.
    """
    assert type(PG0) is np.ndarray, "Period Grid must be an array"

    PcadG0 = np.floor(PG0/keptoy.lc).astype(int)
    nrow   = np.ceil(ncad/PcadG0).astype(int)+1
    remG   = np.round((PG0/keptoy.lc-PcadG0)*nrow).astype(int)

    PG     = (PcadG0 + 1.*remG / nrow)*lc
    return PcadG0,remG,PG

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

def tdpep(t,fm,isStep,P1,P2,twdG):
    """
    Transit-duration - Period - Epoch

    Parameters 
    ---------- 
    fm   : Flux with bad data points masked out.  It is assumed that
           elements of f are evenly spaced in time.
    P1   : First period (cadences)
    P2   : Last period (cadences)
    twdG : Grid of transit durations (cadences)

    Returns
    -------

    rtd : 2-D record array with the following fields at every trial
          (twd,Pcad):
          - noise
          - s2n
          - twd
          - fields in rep
    """
    assert fm.fill_value ==0

    # Determine the grid of periods that corresponds to integer
    # multiples of cadence values
    PcadG = np.arange(P1,P2+1)
    ntwd     = len(twdG)

    rtd = []
    for i in range(ntwd):     # Loop over twd
        twd = twdG[i]

        dM  = mtd(t,fm.filled(),isStep,fm.mask,twd)
        rep = pep(t[0],dM,PcadG)
        r   = np.empty(rep.size, dtype=tddtype)

        for k in epdtype.names:
            r[k] = rep[k]
        r['noise'] = ma.median( ma.abs(dM) )        
        r['twd']   = twd

        rtd.append(r) 

    rtd = np.vstack(rtd)
    rtd['s2n'] = rtd['mean']/rtd['noise']*np.sqrt(rtd['count'])
    return rtd

def pep(t0,dM,PcadG):
    """
    Period-Epoch

    Parameters
    ----------
    t0    : time of first dM[0].
    dM    : depth statistic
    PcadG : Grid of periods (units of cadance)
    """

    func = lambda Pcad: ep(t0,dM,Pcad)
    rep = map(func,PcadG)
    rep = np.hstack(rep)

    return rep

def ep(t0,dM,Pcad0):
    """
    Search in Epoch.

    Parameters
    ----------
    t0   : Time of first cadance.  This is needed to set the epoch.
    dM   : Transit depth estimator
    Pcad : Number of cadances to foldon

    Returns the following information:
    - 'mean'   : Average of the folded columns (does not count masked items)
    - 'count'  : Number of non-masked items.
    - 'epoch'  : epoch maximum mean.
    """
    
    dMW = FFA.XWrap2(dM,Pcad0,pow2=True)
    M   = dMW.shape[0]  # number of rows

    idCol = np.arange(Pcad0,dtype=int)   # id of each column
    idRow = np.arange(M,dtype=int)   # id of each row

    epoch = idCol.astype(float) * lc + t0
    Pcad  = Pcad0 + idRow.astype(float) / (M - 1)

    dMW.fill_value=0
    data = dMW.filled()
    mask = (~dMW.mask).astype(int)

    sumF   = FFA.FFA(data) # Sum of array elements folded on P0, P0 + i/(1-M)
    countF = FFA.FFA(mask) # Number of valid data points
    meanF  = sumF/countF

    rep   = np.empty(M,dtype=epdtype)

    # Take maximum epoch
    idColMa      = meanF.argmax(axis=1)
    rep['mean']  = meanF[idRow,idColMa]
    rep['count'] = countF[idRow,idColMa]
    rep['epoch'] = epoch[idColMa]
    rep['Pcad']  = Pcad

    return rep

def tdmarg(rtd):
    """
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
    iMaTwd = np.argmax(rtd['s2n'],axis=0)
    x      = np.arange(rtd.shape[1])
    rec    = rtd[iMaTwd,x]

    return rec

