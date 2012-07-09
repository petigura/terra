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

def XWrap(x,Pcad0,rem,fill_value=0,pow2=False):
    """
    Extend and wrap array.
    
    Fold array every y indecies.  There will typically be a hanging
    part of the array.  This is padded out.

    Parameters
    ----------

    x     : input
    Pcad0 : Base period
    rem   : The actual period can be longer by rem/nrow
    pow2  : If true, pad out nRows so that it's the next power of 2.

    Return
    ------

    xwrap : Wrapped array.

    """

    ncad = x.size # Number of cadences
    # for some reason np.ceil(ncad/Pcad0) doesn't work!
    nrow = int( np.floor(ncad/Pcad0) +1 )
    nExtend = nrow * Pcad0 - ncad # Pad out remainder of array with 0s.

    if type(x) is np.ma.core.MaskedArray:
        pad = ma.empty(nExtend)
        pad.mask = True
        x = ma.hstack( (x ,pad) )
    else:    
        pad = np.empty(nExtend) 
        pad[:] = fill_value
        x = np.hstack( (x ,pad) )

    xwrap = x.reshape( nrow,-1 )
    idShf = remShuffle(xwrap.shape,rem)
    xwrap = xwrap[idShf]

    return xwrap


def XWrap2(x,P0,fill_value=0,pow2=False):
    """
    Extend and wrap array.
    
    Fold array every y indecies.  There will typically be a hanging
    part of the array.  This is padded out.

    Parameters
    ----------

    x     : input
    P0    : Base period, units of elements
    pow2  : If true, pad out nRows so that it's the next power of 2.

    Return
    ------

    xwrap : Wrapped array.

    """

    ncad = x.size # Number of cadences
    # for some reason np.ceil(ncad/P0) doesn't work!
    nrow = int( np.floor(ncad/P0) +1 )
    nExtend = nrow * P0 - ncad # Pad out remainder of array with 0s.

    if type(x) is np.ma.core.MaskedArray:
        pad = ma.empty(nExtend)
        pad.mask = True
        x = ma.hstack( (x ,pad) )
    else:    
        pad = np.empty(nExtend) 
        pad[:] = fill_value
        x = np.hstack( (x ,pad) )

    xwrap = x.reshape( nrow,-1 )

    if pow2:
        k = np.ceil(np.log2(nrow)).astype(int)
        nrow2 = 2**k
        fill    = ma.empty( (nrow2-nrow,P0) )
        fill[:] = fill_value
        fill.mask=True
        xwrap = ma.vstack([xwrap,fill])

    return xwrap


def remShuffle(shape,rem):
    """
    Remainder shuffle

    For a 2-D array with shape (Pcad0,nrow), this rearanges the
    indecies such that the last row is shifted by rem.  rem can be any
    integer between 0 and Pcad0-1
    
    Parameters
    ----------
    shape : Shape of array to be shuffled.
    rem   : Shift the last row by rem.

    Returns
    -------
    id    : Shuffled indecies.

    """
    nrow,ncol = shape

#    assert type(rem) == int, 'rem must be an integer'
    assert (rem >= 0) & (rem<=ncol), 'rem must be >= 0 and <= ncol '

    irow,icol = np.mgrid[0:nrow,0:ncol]
    colshift  = np.linspace(0,rem,nrow)
    colshift  = np.round(colshift).astype(int)
    for i in range(nrow):
        icol[i] = np.roll(icol[i],-colshift[i])

    return irow,icol


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

    # Determine how many periods we will fold on.
    nPperP0  = 2**np.ceil( np.log2(1.*fm.size/PcadG) )
    nP       = np.sum(nPperP0)

    rtd = np.empty( (ntwd,nP) , dtype=tddtype )
    for i in range(ntwd):     # Loop over twd
        twd = twdG[i]
        rtd['twd'][i] = twd

        dM  = mtd(t,fm.filled(),isStep,fm.mask,twd)
        rep = pep(t[0],dM,PcadG)

        for k in epdtype.names:
            rtd[i][k] = rep[k]

        rtd['noise'][i] = ma.median( ma.abs(dM) )        

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
    
    dMW = XWrap2(dM,Pcad0,pow2=True)
    M   = dMW.shape[0]  # number of rows

    idCol = np.arange(Pcad0,dtype=int)   # id of each column
    idRow = np.arange(M,dtype=int)   # id of each row

    epoch = idCol.astype(float) * lc + t0
    Pcad  = Pcad0 + idRow.astype(float) / (M - 1)

    dMW.fill_value=0
    data = dMW.filled()
    mask = (~dMW.mask).astype(int)

    sumF   = FFA(data) # Sum of array elements folded on P0, P0 + i/(1-M)
    countF = FFA(mask) # Number of valid data points
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

def FFA(XW):
    """
    Fast Folding Algorithm

    Consider an evenly-spaced timeseries of length N.  We can fold it
    on P0, by creating a new array XW, shape = (P0,M), M = N/P0.
    There are M ways to fold XW, yielding the following periods

    P = P0 + i / M - 1

    Where i ranges from 0 to M-1.  Summing all these values requires P
    * M**2 = N**2 / P0 sums.  If M is a power of 2, the FFA eliminates
    some of the redundant summing and requires only N log2 (N/P0)
    sums.

    Algorithm
    ---------
    The columns of XW are shifted and summed in a pairwise fashion.

    - `FFAButterfly` : for a group of size nGroup, `FFAButterfly`
      computes the amount pairwise combinations of rows and the amount
      the second row is shifted.
    - `FFAShiftAdd` : Adds the rows in the manner specified by
      `FFAButterfly`
        
    Parameters
    ----------
    XW : Wrapped array folded on P0.  shape(XW) = (P0,M) and M must be
         a power of 2
    
    Returns
    -------
    XWFS : XW Folded and Summed. 

    References
    ----------
    [1] Staelin (1969)
    [2] Kondratiev (2009)
    """

    # Make sure array is the right shape.
    nRow,P0  = XW.shape
    nStage   = np.log2(nRow)
    assert np.allclose(nStage,np.round(nStage)),"nRow must be power of 2"    
    nStage = int(nStage)

    XWFS = XW.copy()
    for stage in range(1,nStage+1):
        XWFS = FFAShiftAdd(XWFS,stage) 
    return XWFS

def FFAButterfly(stage):
    """
    FFA Butterfly

    The FFA adds pairs of rows A and B. B is shifted.  FFAButterfly
    computes A, B, and the amount by which B is shifted

    Parameters
    ----------
    
    stage : FFA builds up by stages.  Stage 1 shuffles adjacent rows
            (nRowGroup = 2) while stage K shuffles all M = 2**K rows
            (nRowGroup = M).

    """
    nRowGroup = 2**stage

    Arow  = np.empty(nRowGroup,dtype=int)
    Brow  = np.empty(nRowGroup,dtype=int)
    Bshft = np.empty(nRowGroup+2,dtype=int)

    Arow[0::2] = Arow[1::2] = np.arange(0,nRowGroup/2)
    Brow[0::2] = Brow[1::2] = np.arange(nRowGroup/2,nRowGroup)
    Bshft[0::2] = Bshft[1::2] = np.arange(0,nRowGroup/2+1)
    Bshft =  Bshft[1:-1]

    return Arow,Brow,Bshft

def FFAGroupShiftAdd(group0,Arow,Brow,Bshft):
    """
    FFA Shift and Add

    Add the rows of `group` to each other.
    
    Parameters
    ----------

    group0 : Initial group before shuffling and adding. 
             shape(group0) = (M,P0) where M is a power of 2.

    """
    nRowGroup,nColGroup = group0.shape
    group     = np.empty(group0.shape)

    sizes = np.array([Arow.size, Brow.size, Bshft.size])
    assert (sizes == nRowGroup).all() , 'Number of rows in group must agree with butterfly output'

    # Grow group by the maximum shift value
    maxShft = max(Bshft)
    group0 = np.hstack( [group0 , group0[:,: maxShft]] )

    for iRow in range(nRowGroup):
        iA = Arow[iRow]
        iB = Brow[iRow]
        Bs = Bshft[iRow]

        A = group0[iA][:-maxShft] 
        B = group0[iB][Bs:Bs+nColGroup]

        group[iRow] = A + B

    return group 

def FFAShiftAdd(XW0,stage):
    """
    FFA Shift and Add

    Shuffle pairwise add the rows of the FFA data array corresponding
    to stage
    
    Parameters
    ----------
    XW0   : array
    stage : The stage in the FFA.  An integer ranging from 1 to K
            where 2**K = M
            
    Returns
    -------
    XW    : Shifted and added array


    Test Cases
    ----------

    >>> tfind.FFAShiftAdd(eye(4),1)
    >>> array([[ 1.,  1.,  0.,  0.],
               [ 2.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  1.],
               [ 0.,  0.,  2.,  0.]])
    """
    nRow      = XW0.shape[0]
    nRowGroup = 2**stage
    nGroup    = nRow/nRowGroup
    XW        = np.empty(XW0.shape)
    Arow,Brow,Bshft = FFAButterfly(stage)
    for iGroup in range(nGroup):
        start = iGroup*nRowGroup
        stop  = (iGroup+1)*nRowGroup
        sG = slice(start,stop)
        XW[sG] = FFAGroupShiftAdd(XW0[sG],Arow,Brow,Bshft)

    return XW

