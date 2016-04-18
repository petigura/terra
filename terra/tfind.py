"""
Transit finder.

Evaluate a figure of merit at each point in P,epoch,tdur space.
"""
import itertools

import scipy
from scipy import ndimage as nd
import numpy as np
from numpy import ma
import pandas as pd
import h5py

from FFA import FFA_cext as FFA
import config
from utils import h5plus
from keptoy import P2a,a2tdur
from utils.h5plus import h5F

# dtype of the record array returned from ep()
epnames = ['mean','count','t0cad','Pcad']
epdtype = zip(epnames,[float]*len(epnames) )
epdtype = np.dtype(epdtype)

# dtype of the record array returned from tdpep()
tdnames = epnames + ['noise','s2n','twd','t0']
tddtype = zip(tdnames,[float]*len(tdnames))
tddtype = np.dtype(tddtype)

class Grid(object):
    def __init__(self, t, fm):
        """
        Initialize a grid object
        """
        self.t = t 
        self.fm = fm

    def set_parL(self,parL):
        """
        Set grid search parameters

        Args:
             parL (list) : List of dictionaries each with the following keys:
                 - Pcad1 : Lower bound of period search (integer # of cadences)
                 - Pcad2 : Upper bound + 1
                 - twdG  : Range of trial durations to search over 

        """

        names = 'P1 P2 twdG'.split()
        print pd.DataFrame(parL)[names]
        self.parL = parL
        
    def periodogram(self,mode='std'):
        """
        Run the transit finding periodogram
        
        mode : What type of periodogram to run?
          - 'max' runs pgram_max transit finder. Clips top two SES
          - 'bls' my implementation of BLS computes folds and then evaluates SNR
          - 'ffa' SES then fold
          - 'fm' foreman mackey algo

        Returns
        -------
        pgram : Pandas dataframe with mean, count, t0cad, Pcad, noise,
                s2n, twd, t0

        """
        if mode=='max':
            pgram = map(self._pgram_max,self.parL)
            pgram = np.hstack(pgram)
            pgram = pd.DataFrame(pgram)
        if mode=='ffa':
            pgram = map(self._pgram_ffa,self.parL)
            pgram = pd.DataFrame(np.hstack(pgram))
        if mode=='bls':
            pgram = map(self._pgram_bls,self.parL)
            pgram = np.hstack(pgram)
            pgram = pd.DataFrame(pgram)
            pgram['t0'] = self.t[0] + (pgram['col']+pgram['twd']/2)*config.lc
        if mode=='fm':
            pgram = map(self._pgram_fm,self.parL)
            pgram = np.hstack(pgram)
            pgram = pd.DataFrame(pgram)
            pgram['t0'] = self.t[0] + (pgram['col']+pgram['twd']/2)*config.lc
            pgram['s2n'] = pgram['depth_2d'] * np.sqrt(pgram['depth_ivar_2d'])
            pgram['mean'] = pgram['depth_2d']
            pgram['noise'] = 1/np.sqrt(pgram['depth_ivar_2d'])

        self.pgram = pgram
        return pgram

    def _pgram_ffa(self,par):
        rtd = tdpep(self.t,self.fm,par)
        r = tdmarg(rtd)
        return r

    def _pgram_max(self,par):
        pgram = pgram_max(self.t,self.fm,par)
        return pgram

    def _pgram_bls(self,par):
        pgram = bls(self.t,self.fm,par)
        return pgram

    def _pgram_fm(self,par):
        pgram = foreman_mackey(self.t,self.fm,par)
        return pgram

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

def mtd(fm,twd):
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

    assert isinstance(twd,int),"Box width most be integer number of cadences"

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

def tdpep(t,fm,par):
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
    PcadG = np.arange(par['Pcad1'],par['Pcad2'])
    twdG = par['twdG']
    assert fm.fill_value==0
    # Determine the grid of periods that corresponds to integer
    # multiples of cadence values

    ntwd  = len(twdG)

    rtd = []
    for i in range(ntwd):     # Loop over twd
        twd = twdG[i]
        dM  = mtd(fm,twd)

        func = lambda Pcad: ep(dM,Pcad)
        rep = map(func,PcadG)
        rep = np.hstack(rep)
        r   = np.empty(rep.size, dtype=tddtype)
        for k in epdtype.names:
            r[k] = rep[k]
        r['noise'] = ma.median( ma.abs(dM) )
        r['twd']   = twd
        r['t0']    = r['t0cad']*config.lc + t[0]        
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
    
    t0cad,Pcad,meanF,countF = fold_ffa(dM,Pcad0)
    rep = epmarg(t0cad,Pcad,meanF,countF)
    return rep

def fold_ffa(dM,Pcad0):
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

def pgram_max(t,fm,par):
    """
    Periodogram: Check max values

    Computes s2n for range of P, t0, and twd. However, for every
    putative transit, we evaluate the transit depth having cut the
    deepest transit and the second deepest transit. We require that
    the mean depth after having cut the test max two values not be too
    much smaller. Good at removing locations with 2 outliers.

    Parameters 
    ----------
    t : t[0] provides starting time
    fm : masked array with fluxes
    par : dict with following keys
          - Pcad1 (lower period limit)
          - Pcad2 (upper period limit)
          - twdG (grid of trial durations to compute)

    Returns
    -------
    pgram : Record array with following fields

    """
    ncad = fm.size
    PcadG = np.arange(par['Pcad1'],par['Pcad2'])
    get_frac_Pcad = lambda P : np.arange(P,P+1,1.0*P / ncad)
    PcadG = np.hstack(map(get_frac_Pcad,PcadG))
    twdG = par['twdG']
    
    icad = np.arange(ncad)

    dtype_pgram = [
        ('Pcad',float),
        ('twd',float),
        ('s2n',float),
        ('c',float),
        ('mean',float),
        ('t0',float),
        ('noise',float),
        ]

    pgram = np.zeros( (len(twdG),len(PcadG)),dtype=dtype_pgram)

    # dtype of the record array returned from tdpep()
    dtype_temp = [
        ('c',int),
        ('mean',float),
        ('s2n',float),
        ('col',int),
        ('t0',float)
    ] 

    # Loop over different transit durations
    for itwd,twd in enumerate(twdG):
        res = foreman_mackey_1d(fm,twd)    
        dM = ma.masked_array(
            res['depth_1d'],
            ~res['good_trans'].astype(bool),
            fill_value=0
            )

        # Compute noise (robust, based on MAD) on twd timescales
        noise = ma.median( ma.abs(dM) )[0] * 1.5
        pgram[itwd,:]['noise'] = noise
        pgram[itwd,:]['twd'] = twd
        for iPcad,Pcad in enumerate(PcadG):
            # Compute row and columns for folded data
            row,col = wrap_icad(icad,Pcad)
            
            ncol = np.max(col) + 1
            nrow = np.max(row) + 1
            icol = np.arange(ncol)

            # Shove data and mask into appropriate positions
            data = np.zeros((nrow,ncol))
            mask = np.ones((nrow,ncol)).astype(bool)
            data[row,col] = dM.data
            mask[row,col] = dM.mask

            # Sum along columns (clipping top 0, 1, 2 values)
            datasum,datacnt = cumsum_top(data,mask,2)
            # datasum[-1] are the is the summed columns having not
            # clipped any values. Update results array. For t0, add
            # half the transit with because column index corresponds
            # in ingress
            s = datasum[-1,:]
            c = datacnt[-1,:]
            r = np.zeros(ncol,dtype=dtype_temp)
            r['s2n'] = -1 
            r['mean'] = s/c
            r['s2n'] = s / np.sqrt(c) / noise            
            r['c'][:] = c
            r['col'] = icol
            r['t0'] = ( r['col'] + twd / 2.0) * config.lc + t[0] 

            # Compute mean transit depth after removing the deepest
            # transit, datacnt[-2], and the second deepest transit,
            # datacnt[-3]. The mean transit depth must be > 0.5 it's
            # former value. Also, require 3 transits.
            mean_clip1 = datasum[-2] / datacnt[-2]
            mean_clip2 = datasum[-3] / datacnt[-3]
            b = (
                (mean_clip1 > 0.5 * r['mean'] ) & 
                (mean_clip2 > 0.5 * r['mean']) & 
                (r['c'] >= 3)
            )

            if ~np.any(b):
                continue 

            rcut = r[b]
            rmax = rcut[np.argmax(rcut['s2n'])]
            names = ['mean','s2n','c','t0']
            for n in names:
                pgram[itwd,iPcad][n] = rmax[n]
            pgram[itwd,iPcad]['Pcad'] = Pcad
            
    # Compute the maximum return twd with the maximum s2n
    pgram = pgram[np.argmax(pgram['s2n'],axis=0),np.arange(pgram.shape[1])]
    return pgram

def bls(t,fm,par):
    """
    """
    ncad = fm.size
    PcadG = np.arange(par['Pcad1'],par['Pcad2'])
    get_frac_Pcad = lambda P : np.arange(P,P+1,1.0*P / ncad)
    PcadG = np.hstack(map(get_frac_Pcad,PcadG))

    data = fm.data
    mask = fm.mask.astype(int)
    icad = np.arange(fm.size)

    twd1 = par['twdG'][0]
    twd2 = par['twdG'][-1]

    dtype = [('col',int),('twd',float),('s2n',float),('Pcad',float),('noise',float),('mean',float)]
    pgram = np.zeros(PcadG.size,dtype)
    for i,Pcad in enumerate(PcadG):
        row,col = wrap_icad(icad,Pcad)
        ccol,scol,sscol = fold.fold_col(data,mask,col)
        ncol = np.max(col) + 1
        r = pgram[i]
        r['s2n'],r['twd'],r['col'],r['mean'],r['noise'] = fold.bls(
            ccol,scol,sscol,ncol,twd1,twd2
        )
        r['Pcad'] = Pcad
    pgram['mean'] *= -1
    return pgram

dtype = [
    ('phic_same', float), 
    ('phic_variable', float),
    ('depth_2d', float),
    ('depth_ivar_2d', float),
    ('nind', float),
    ('col', int),
    ('itwd', int),
]

dtype_fm_max_res = np.dtype(dtype)


dtype = [
    ('phic_same', float), 
    ('phic_variable', float),
    ('depth_2d', float),
    ('depth_ivar_2d', float),
    ('nind', float),
    ('col', int),
    ('itwd', int),
    ('Pcad',float),
    ('twd', int),
]

dtype_fm_res = np.dtype(dtype)


def foreman_mackey(t,fm,par):
    """
    """
    ncad = fm.size
    Pcad1 = par['Pcad1']
    Pcad2 = par['Pcad2']
    twdG = par['twdG']
    alpha = 1200.0

    PcadG = np.arange(Pcad1, Pcad2)
    get_frac_Pcad = lambda P : np.arange(P,P+1,1.0*P / ncad)
    PcadG = np.hstack(map(get_frac_Pcad,PcadG))
    nPcad = PcadG.size

    res_1d = map(lambda x : foreman_mackey_1d(fm,x), twdG)
    res_1d = np.vstack(res_1d)


    pgram = np.zeros(nPcad,dtype=dtype_fm_res)
    
    for i,Pcad in enumerate(PcadG):
        res = fold.forman_mackey_max(
            Pcad, 
            alpha, 
            res_1d['good_trans'],
            res_1d['dll_1d'],
            res_1d['depth_1d'],
            res_1d['depth_ivar_1d']
        )

        for iname,name in enumerate(dtype_fm_max_res.names):
            pgram[i][name] = res[iname]

        pgram[i]['Pcad'] = Pcad
#        pgram[i]['twd'] = twdG[pgram[i]['itwd']]

    return pgram

def foreman_mackey(t,fm,par):
    """
    """
    ncad = fm.size
    Pcad1 = par['Pcad1']
    Pcad2 = par['Pcad2']
    twdG = par['twdG']
    ntwd = len(twdG)
    alpha = 1200.0

    PcadG = np.arange(Pcad1, Pcad2)
    get_frac_Pcad = lambda P : np.arange(P,P+1,1.0*P / ncad)
    PcadG = np.hstack(map(get_frac_Pcad,PcadG))
    nPcad = PcadG.size

    icad = np.arange(fm.size)
    data = fm.data
    mask = fm.mask.astype(int)

    res_1d = map(lambda x : foreman_mackey_1d(fm,x), twdG)
    res_1d = np.vstack(res_1d)

    pgram = np.zeros(nPcad,dtype=dtype_fm_res)

    dtype = [
        ('col', int), 
        ('phic_same',float),
        ('phic_variable',float),
        ('depth_2d',float), 
        ('depth_ivar_2d',float),
        ('nind',int),
        ('idx',int),
        ('twd',int),
    ]
    
    for i,Pcad in enumerate(PcadG):
        row,col = wrap_icad(icad,Pcad)
        ncol = np.max(col) + 1
        res_temp = np.zeros((ntwd,ncol),dtype=dtype)
        res_temp['phic_same'] -= np.inf
        res_temp['phic_variable'] -= np.inf

        for itwd,twd in enumerate(twdG):
            col, phic_same, phic_variable, depth_2d, depth_ivar_2d, nind = \
                fold.forman_mackey(
                    Pcad, 
                    alpha, 
                    res_1d[itwd]['good_trans'], 
                    res_1d[itwd]['dll_1d'], 
                    res_1d[itwd]['depth_1d'], 
                    res_1d[itwd]['depth_ivar_1d'], 
                )


            res_temp[itwd]['col'] = col
            res_temp[itwd]['phic_same'] = phic_same
            res_temp[itwd]['phic_variable'] = phic_variable
            res_temp[itwd]['depth_2d'] = depth_2d
            res_temp[itwd]['depth_ivar_2d'] = depth_ivar_2d
            res_temp[itwd]['nind'] = nind
            res_temp[itwd]['twd'] = twd

        res_temp = res_temp.flatten()
        res_temp = res_temp[
            (res_temp['depth_2d'] > 0.0 ) &
            (res_temp['phic_same'] > res_temp['phic_variable']) &
            (res_temp['nind'] >= 2)
            ]

        if res_temp.size==0:
            continue 

        res_max = res_temp[res_temp['phic_same'].argmax()]

        pgram[i]['phic_same'] = res_max['phic_same']
        pgram[i]['phic_variable'] = res_max['phic_variable']
        pgram[i]['depth_2d'] = res_max['depth_2d']
        pgram[i]['depth_ivar_2d'] = res_max['depth_ivar_2d']
        pgram[i]['nind'] = res_max['nind']
        pgram[i]['col'] = res_max['col']
        pgram[i]['twd'] = res_max['twd']
        pgram[i]['Pcad'] = Pcad        

    return pgram

def foreman_mackey_1d(fm,twd):
    assert fm.fill_value==0,'fill_value must = 0'
    assert np.sum(np.isnan(fm.compressed()))==0,'mask out nans'

    dtype = [
        ('good_trans', int),
        ('depth_1d', float), 
        ('depth_ivar_1d', float), 
        ('dll_1d', float), 
    ]

    ncad = len(fm)
    fmfilled = fm.filled()

    # Compute inverse varience
    ivar = 1.0 / np.median(np.diff(fm.compressed()) ** 2)
    res = np.zeros(ncad,dtype=dtype)

    for cad1 in range(ncad):
        cad2 = cad1 + twd
        data = fmfilled[cad1:cad2]
        mask = fm.mask[cad1:cad2]
        s = np.sum(data)
        c = np.sum(~mask)
        m = s / c
        res['depth_1d'][cad1] = -1.0 * m 
        ll0 = -0.5 * np.sum(data**2) * ivar 
        ll = -0.5 * np.sum( (data - m )**2 ) * ivar 
        res['dll_1d'][cad1] = ll - ll0
        res['depth_ivar_1d'][cad1] = ivar * c
        
        if c > twd / 2:
            res['good_trans'][cad1] = 1

    return res

def get_frac_Pcad(P):
    """
    Return all the fractional periods between P1 and P1 + 1
    """
    #    step = 0.25 * P1 * twd / tbase
    step = 1./P
    return np.arange(P,P+1,step)


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


def _periodogram_parameters_segment(P1, P2, tbase, Rstar=1, Mstar=1, 
                                    ftdur=[0.5,1.5]):
    """Compute periodogram parameters for a single segment"""

    fLastOff = 0.25 # Choose period resolution such that the transits
                    # line up to better than fLastOff * tdur

    Plim  = np.array([P1,P2])

    alim = P2a(Plim,Mstar=Mstar )
    tdurlim = a2tdur( alim , Mstar=Mstar,Rstar=Rstar ) * ftdur

    qlim = tdurlim / Plim
    qmi,qma =  min(qlim)*min(ftdur) , max(qlim)*max(ftdur)
    fmi = 1 / P2
    df = fmi * fLastOff * min(tdurlim)  / tbase 
    nf = int((1/P1 - 1/P2) / df)
    farr = fmi + np.arange(nf)*df        
    delTlim = np.round(tdurlim / config.lc).astype(int)
    Pc = np.sqrt(P1*P2) # central period.
    nb = int(Pc / config.lc)

    Pcad1 = int(P1/config.lc)
    Pcad2 = int(P2/config.lc)

    twdG = []
    delT = delTlim[0]
    while delT < delTlim[1]:
        twdG.append( int(round(delT) ) )
        delT *= 1.33

    twdG.append(delTlim[1])

    d = dict( qmi=qmi, qma=qma, fmi=fmi, nf=nf, df=df, farr=farr,nb=nb ,
              delT1=delTlim[0], delT2=delTlim[1], Pcad1=Pcad1, Pcad2=Pcad2,
              twdG=twdG)
    return d

def periodogram_parameters(P1, P2, tbase, nseg, Rstar=1.0, Mstar=1.0, 
                           ftdur=[0.5,1.5]):
    """
    Periodogram Parameters.

    Args:
        P1 (float): Minimum period
        P2 (float): Maximum period
        nseg (int): Use the following number of segments 
        Rstar (Optional[float]) : Stellar radius [Rsun].
        Mstar (Optional[float]) : Stellar mass [Msun]. Used with Rstar to 
             compute expected duration.
        ftdur (Optional[list]) : Fraction of expected maximum tranit duration 
             to search over.
    """

    # Split the periods into logrithmically spaced segments
    PlimArr = np.logspace( np.log10(P1) , np.log10(P2),nseg+1  )
    dL = []
    for i in range(nseg):
        P1 = PlimArr[i]
        P2 = PlimArr[i+1]
        d = _periodogram_parameters_segment(
            P1,P2,tbase,Rstar=Rstar,Mstar=Rstar,ftdur=ftdur
            )
        d['P1'] = P1
        d['P2'] = P2
        dL.append(d)

    return dL

def cumsum_top(data,mask,k):
    """

    Take a data and mask array with shape = (m,n)

    For each column of data, sort the values (ignoring values that are
    masked out). Then compute the sum along columns of the m-k
    smallest values, then the m-k+1 smallest values until all m rows
    are summed.

    Parameters 
    ----------
    data : ndim = 2,
    mask : True if entry is included, False if not
    """
    

    # masked values are filled with -infs (so they don't enter into the sort)
    nrow,ncol = data.shape
    nrowout = k + 1

    datainf = data.copy() 
    datazero = data.copy()

    datainf[mask] = -np.inf 
    datazero[mask] = 0.0

    icol = np.arange(data.shape[1])
    srow = np.argsort(datainf,axis=0) # Indecies of sorted rows
    
    datazero = datazero[srow,icol]
    mask = mask[srow,icol]
    cnt = (~mask).astype(int) # Counter. 1 if it's a valid point

    # Substitue the sum of all the higher rows in to the nrowout row
    # of output arrays
    datazero[-nrowout,:] = np.sum(datazero[:-k],axis=0)
    cnt[-nrowout,:] = np.sum(cnt[:-k],axis=0)
    
    datasum = np.cumsum(datazero[-nrowout:],axis=0)
    # Count number of masked elements
    datacnt = np.cumsum(cnt[-nrowout:],axis=0) 
    return datasum,datacnt

def test_cumsum_top():
    np.set_printoptions(precision=1)
    nrow,ncol = 4,10
    
    arr = np.ones((nrow,ncol))
    arr[0,2] = np.nan
    arr[-1,-3:] = np.nan
    arr[:,3] = 2
    arr[1,5] = 8
    arr[2,5] = 10

    print "Input array"
    print arr 

    mask = np.isnan(arr)
    k = 2
    print "cumsum arrays"
    datasum,datacnt  = cumsum_top(arr,mask,k)
    print datasum.astype(int)
    print datacnt.astype(int)
    print "Mean value"
    s =  str(datasum.astype(float)/(datacnt.astype(float)))
    sL = s.split('\n')
    sL[0] += ' <- Consistent depth'
    sL[2] += ' <- 2 outliers drive mean'
    print "\n".join(sL)
    

def wrap_icad(icad,Pcad):
    """
    rows and column identfication to each one of the
    measurements in df

    Parameters
    ----------
    icad : Measurement number starting with 0
    Pcad : Period to fold on
    """

    row = np.floor( icad / Pcad ).astype(int)
    col = np.floor(np.mod(icad,Pcad)).astype(int)
    return row,col

