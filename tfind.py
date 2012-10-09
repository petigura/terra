"""
Transit finder.

Evaluate a figure of merit at each point in P,epoch,tdur space.

Future work: implement a general linear function that works along folded columns.  Right now I'm doing a weighted mean. 

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

import config
import prepro
import h5plus
import h5py

# dtype of the record array returned from ep()
epnames = ['mean','count','t0cad','Pcad']
epdtype = zip(epnames,[float]*len(epnames) )
epdtype = np.dtype(epdtype)

# dtype of the record array returned from tdpep()
tdnames = epnames + ['noise','s2n','twd','t0']
tddtype = zip(tdnames,[float]*len(tdnames))
tddtype = np.dtype(tddtype)

class Grid(h5plus.File):
    # Default period limits.  Can change after instaniation.
    P1 = int(np.floor(config.P1/keptoy.lc))
    P2 = int(np.floor(config.P2/keptoy.lc))
    cut = 5e3

    def __init__(self,*args):
        h5plus.File.__init__(self,args[0])
        if len(args) is 2:
            hlc      = h5py.File(args[1],mode='r')
            self['mqcal'] = hlc['mqcal'][:]

    def grid(self):
        """
        Run the grid search
        """
        lc = self['mqcal'][:]
        fm  = ma.masked_array(lc['fcal'],lc['fmask'],fill_value=0,copy=True)
        rtd = tdpep(lc['t'],fm,self.P1,self.P2,config.twdG)
        r   = tdmarg(rtd)
        self['RES'] = r

    def itOutRej(self):
        """
        Run the iterative outlier rejection

        """
        it = 0 
        done = False
        res0 = self['RES'][:]
        lc0  = self['mqcal'][:]

        while done is False:
            cad = lc0['cad']
            c = cadCount(cad,res0)
            bout = c > config.maCadCnt
            nout = c[bout].size
            print "%s: it%i maCadCnt = %i, %i outliers" % (self,it,max(c),nout)
            if (nout==0 ) or (it > 2):
                done = True
            else:
                lc1   = lc0.copy()

                # Update the mask
                bout = nd.convolve(bout.astype(float),np.ones(20) ) >  0
                lc1['fmask']  = lc1['fmask'] | bout
                fm  = ma.masked_array(lc1['fcal'],lc1['fmask'],fill_value=0)
                
                # Rerun the grid search 
                rtd  = tdpep(lc1['t'],fm,self.P1,self.P2,twdG)
                res1 = tdmarg(rtd)
                self['RES'][:] = res1

                gpname = 'it%i' % it
                grp = self.create_group(gpname)

                print "Storing Previous Iteration to Group %s" % gpname
                grp['RES'] = res0 
                grp['mqcal'] = lc0

                lc0  = lc1.copy()
                res0 = res1.copy()

            it +=1

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

def mtd(t,fm,twd):
    """
    Mean Transit Depth

    Convolve time series with our locally detrended matched filter.  

    Parameters
    ----------
    t      : time series.
    fm     : masked flux array.  masked regions enter into the average
             with 0 weight.
    twd    : Width of kernel in cadances

    Notes
    -----
    Since a single nan in the convolution kernel will return a nan, we
    interpolate the entire time series.  We see some edge effects

    """
    fm = fm.copy()
    fm.fill_value = 0
    w = (~fm.mask).astype(int) # weights either 0 or 1
    f = fm.filled()

    pad = np.zeros(twd)
    f = np.hstack([pad,f,pad])
    w = np.hstack([pad,w,pad])

    assert (np.isnan(f)==False).all() ,'mask out nans'
    kern = np.ones(twd,float)

    ws = np.convolve(w*f,kern,mode='same') # Sum of points in bin
    c = np.convolve(w,kern,mode='same')    # Number of points in bin

    # Number of good points before, during, and after transit
    bc = c[:-2*twd]
    tc = c[twd:-twd]
    ac = c[2*twd:]

    # Weighted sum of points before, during and after transit
    bws = ws[:-2*twd]
    tws = ws[twd:-twd]
    aws = ws[2*twd:]
    dM = 0.5*(bws/bc + aws/ac) - tws/tc
    dM = ma.masked_invalid(dM)
    dM.fill_value =0

    # Require 0.5 of the points before, during and after transit to be good.
    gap = (bc < twd/2) | (tc < twd/2) | (ac < twd/2)
    dM.mask = dM.mask | gap

    return dM

def tdpep(t,fm,P1,P2,twdG):
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
    ntwd  = len(twdG)

    rtd = []
    for i in range(ntwd):     # Loop over twd
        twd = twdG[i]
        dM  = mtd(t,fm,twd)

        # Compute MES over period grid
        func = lambda Pcad: ep(dM,Pcad)
        rep = map(func,PcadG)
        rep = np.hstack(rep)

        r   = np.empty(rep.size, dtype=tddtype)
        for k in epdtype.names:
            r[k] = rep[k]

        r['noise'] = ma.median( ma.abs(dM) )        
        r['twd']   = twd
        r['t0']    = r['t0cad']*lc + t[0]
        rtd.append(r) 

    rtd = np.vstack(rtd)
    rtd['s2n'] = rtd['mean']/rtd['noise']*np.sqrt(rtd['count'])
    return rtd

def ep(dM,Pcad0):
    """
    Search from Pcad0 to Pcad0+1

    Parameters
    ----------
    dM    : Transit depth estimator
    Pcad0 : Number of cadances to foldon
 
    Returns the following information:
    - 'mean'   : Average of the folded columns (does not count masked items)
    - 'count'  : Number of non-masked items.
    - 't0cad'  : epoch of maximum MES (cadences)
    - 'Pcad'   : Periods that the FFA computed MES 
    """
    
    t0cad,Pcad,meanF,countF = fold(dM,Pcad0)
    rep = epmarg(t0cad,Pcad,meanF,countF)
    return rep

def fold(dM,Pcad0):
    """
    Fold on M periods from Pcad0 to Pcad+1 where M is N / Pcad0
    rounded up to the nearest power of 2.

    Parameters
    ----------
    dM    : Transit depth estimator
    Pcad0 : Number of cadances to foldon
 
    Returns
    -------
    t0cad : Array with the trial epochs  [0, ...,  P0]
    Pcad  : Array with the trial periods [P0, ..., P0]

    meanF  : Average of the folded columns (does not count masked items)
    countF : Number of non-masked items.    


    Notes
    -----
    meanF and coundF have the following shape:

        ep1 ep2 ep3 ... epP1
        --- --- ---     ----
    P0 |  .   .   .       .
    P1 |  .   .   .       .
    P2 |  .   .   .       .
    .  |            .
    .  |              .
    .  |                .
    P3 |  .   .   .       .
    """

    dMW = FFA.XWrap2(dM,Pcad0,pow2=True)
    M   = dMW.shape[0]  # number of rows

    idCol = np.arange(Pcad0,dtype=int)   # id of each column
    idRow = np.arange(M,dtype=int)       # id of each row

    t0cad = idCol.astype(float)
    Pcad  = Pcad0 + idRow.astype(float) / (M - 1)

    dMW.fill_value=0
    data = dMW.filled()
    mask = (~dMW.mask).astype(int)

    sumF   = FFA.FFA(data) # Sum of array elements folded on P0, P0 + i/(1-M)
    countF = FFA.FFA(mask) # Number of valid data points
    meanF  = sumF/countF
    return t0cad,Pcad,meanF,countF

def epmarg(t0cad,Pcad,meanF,countF):
    """
    Epoch Marginalize
    
    Reduce the M x Pcad0 array returned by the FFA to a M length
    array.  For all the trial periods choose the epoch,mean,count of
    the maximum mean.
    """
    idColMa      = np.nanargmax(meanF,axis=1)
    idRow        = np.arange(Pcad.size,dtype=int)
    rep          = np.empty(Pcad.size,dtype=epdtype)
    rep['mean']  = meanF[idRow,idColMa]
    rep['count'] = countF[idRow,idColMa]
    rep['t0cad'] = t0cad[idColMa]
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

def cadCount(cad,res):
    """
    res : record array with t0cad and Pcad fields
    """
    cadmi,cadma = cad[0],cad[-1]
    cadLL = []
    for t0cad,Pcad in zip(res['t0cad'],res['Pcad']):
        nT   = np.floor((cad.size - t0cad ) / Pcad) + 1
        cadL = np.round(t0cad + np.arange(nT)*Pcad)+cadmi
        cadLL.append(cadL)
    cadLL = np.hstack(cadLL)
    c,b = np.histogram(cadLL,bins=np.linspace(cadmi,cadma+1,cad.size+1))
    return c
