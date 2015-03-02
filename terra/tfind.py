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
from FFA import fold
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

def read_hdf(kwargs):
    with h5F(kwargs) as h5:
        lc  = h5['/pp/cal'][:]

    f = lc[ kwargs['fluxField'] ]
    mask = lc[ kwargs['fluxMask'] ] 

    t = lc['t']
    fm = ma.masked_array(f,mask,fill_value=0,copy=True)
    grid = Grid(t,fm)
    return grid

class Grid(object):
    def __init__(self,t,fm):
        self.t = t 
        self.fm = fm
    def set_parL(self,parL):
        """
        Set grid search parameters

        Parameters
        ----------
        parL : List of dictionaries each with the following keys:
               - Pcad1 : Lower bound of period search (integer # of cadences)
               - Pcad2 : Upper bound + 1
               - twdG  : Range of trial durations to search over
        """
        print pd.DataFrame(parL)
        self.parL = parL
        
    def periodogram(self,mode='std'):
        if mode=='std':
            pgram = map(self.pgram_std,self.parL)
            pgram = pd.concat(pgram)
            pgram = pgram.sort(['Pcad','s2n'])
            pgram = pgram.groupby('Pcad',as_index=False).last()
        if mode=='ffa':
            pgram = map(self.pgram_ffa,self.parL)
            pgram = np.hstack(pgram)
            pgram = pd.DataFrame(pgram)
        if mode=='bls':
            pgram = map(self.pgram_bls,self.parL)
            pgram = np.hstack(pgram)
            pgram = pd.DataFrame(pgram)
            pgram['t0'] = self.t[0] + (pgram['col']+pgram['twd']/2)*config.lc
        if mode=='fm':
            pgram = map(self.pgram_fm,self.parL)
            pgram = np.hstack(pgram)
            pgram = pd.DataFrame(pgram)
            pgram['t0'] = self.t[0] + (pgram['col']+pgram['twd']/2)*config.lc
            pgram['s2n'] = pgram['depth_2d'] * np.sqrt(pgram['depth_ivar_2d'])
            pgram['mean'] = pgram['depth_2d']
            pgram['noise'] = 1/np.sqrt(pgram['depth_ivar_2d'])


        self.pgram = pgram
        return pgram

    def pgram_ffa(self,par):
        rtd = tdpep(self.t,self.fm,par)
        r = tdmarg(rtd)
        return r

    def pgram_std(self,par):
        pgram = tdpep_std(self.t,self.fm,par)
        return pgram

    def pgram_bls(self,par):
        pgram = bls(self.t,self.fm,par)
        return pgram

    def pgram_fm(self,par):
        pgram = foreman_mackey(self.t,self.fm,par)
        return pgram


    def to_hdf(self,groupname,kwargs):
        with h5F(kwargs) as h5:
            it0 = h5.create_group(groupname)
            g = h5[groupname]
            g['RES'] = np.array(self.pgram.to_records(index=False))
            print "saving periodogram to %s[%s]" % (h5.filename,groupname)


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

def tdpep_std(t,fm,par):
    """
    """
    ncad = fm.size
    PcadG = np.arange(par['Pcad1'],par['Pcad2'])
    get_frac_Pcad = lambda P : np.arange(P,P+1,1.0*P / ncad)
    PcadG = np.hstack(map(get_frac_Pcad,PcadG))

    icad = np.arange(ncad)

    data = list(itertools.product(par['twdG'],PcadG))
    pgram = pd.DataFrame(data=data,columns='twd Pcad'.split())
    pgram['s2n'] = 0.0
    pgram['c'] = 0.0
    pgram['mean'] = 0.0
    pgram['std'] = 0.0
    pgram['noise'] = 0.0
    pgram['t0'] = 0.0
    pgram['colmax'] = -1

    idx = 0 

    # dtype of the record array returned from tdpep()
    dtype = [
        ('c',int),
        ('mean',float),
        ('std',float),
        ('s2n',float),
        ('col',int),
        ('t0',float)] 

    for twd in par['twdG']:
        dM = mtd(fm,twd)
        dM.fill_value=0
        noise = ma.median( ma.abs(dM) )[0]

        for Pcad in PcadG:
            row,col = fold.wrap_icad(icad,Pcad)
            ncol = np.max(col) + 1 # Starts from 0
            
            r = np.empty(ncol,dtype)
            c,s,ss = fold.fold_col(dM.data,dM.mask.astype(int),col)

            # Compute first and second moments
            r['mean'] = s/c
            r['std'] = np.sqrt( (c*ss-s**2) / (c * (c - 1)))
            r['s2n'] = s / np.sqrt(c) / noise            
            r['c'] = c
            r['col'] = np.arange(ncol) 
            r['t0'] = r['col'] * config.lc + t[0]

            # Non-linear part.
            # - Require 3 or more transits
            # - Require Consistency among transits
            b = (r['c'] >= 3) & (r['std'] < 5 * noise)
            if np.any(b):
                r = r[b] # Cut the columns that don't pass

                imax = np.argmax(r['s2n'])
                pgram.at[idx,'noise'] = noise
                pgram.at[idx,'colmax'] = r['col'][imax]
                for k in 'c mean std s2n t0'.split():
                    pgram.at[idx,k] = r[k][imax]


            idx+=1

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
        row,col = fold.wrap_icad(icad,Pcad)
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
        row,col = fold.wrap_icad(icad,Pcad)
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


def pgramPars(P1,P2,tbase,Rstar=1,Mstar=1,ftdur=[0.5,1.5]  ):
    """
    Periodogram Parameters.

    P1  - Minimum period
    P2  - Maximum period
    Rstar - Guess of stellar radius 
    Mstar - Guess of stellar mass (used with Rstar to comput expected duration
    ftdur - Fraction of expected maximum tranit duration to search over.
    """

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

def pgramParsSeg(P1,P2,tbase,nseg,Rstar=1,Mstar=1,ftdur=[0.5,1.5]):
    # Split the periods into logrithmically spaced segments
    PlimArr = np.logspace( np.log10(P1) , np.log10(P2),nseg+1  )
    dL = []
    for i in range(nseg):
        P1 = PlimArr[i]
        P2 = PlimArr[i+1]
        d = pgramPars(P1,P2,tbase,Rstar=Rstar,Mstar=Rstar,ftdur=ftdur)
        d['P1'] = P1
        d['P2'] = P2
        dL.append(d)

    return dL

