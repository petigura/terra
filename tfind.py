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

#from keptoy import *
import keptoy
import keplerio
import FFA_cext as FFA
# import FBLS
# import FBLS_cext as FBLS
# import FBLS_cy

from config import *

import config
import h5plus
import h5py
import tval

# dtype of the record array returned from ep()
epnames = ['mean','count','t0cad','Pcad']
epdtype = zip(epnames,[float]*len(epnames) )
epdtype = np.dtype(epdtype)

# dtype of the record array returned from tdpep()
tdnames = epnames + ['noise','s2n','twd','t0']
tddtype = zip(tdnames,[float]*len(tdnames))
tddtype = np.dtype(tddtype)


# Default period limits.  Can change after instaniation.
P1 = int(np.floor(config.P1/keptoy.lc))
P2 = int(np.floor(config.P2/keptoy.lc))
maCadCnt = int( np.floor(config.maCadCnt) )
cut = 5e3

def grid(h5):
    """
    Run the grid search

    Parameters
    ----------
    h5 : an h5plus instance
         P1,P2 must be attributes
    """

    lc  = h5['/pp/mqcal'][:]


    P1 = h5.attrs['P1_FFA']
    P2 = h5.attrs['P2_FFA']
    f     = lc[ h5.attrs['fluxField'] ]
    mask = lc[ h5.attrs['fluxMask'] ] 
    t    = lc['t']

    print P1,P2
    print "itOutRej first run. Creating it0"

    it0 = h5.create_group('it0')
    it0 = h5['it0']


    fm    = ma.masked_array(f,mask,fill_value=0,copy=True)
    PcadG = np.arange(P1,P2)
    rtd   = tdpep(t,fm,PcadG,config.twdG)
    r     = tdmarg(rtd)
    it0['RES']   = r

def itOutRej(h5):
    """
    Iterative outlier rejection.

    Parameters
    ----------
    h5 : an h5plus instance
         P1,P2 must be attributes

    Notes
    -----
    itOutRej can exit with 3 status:
    1. Large peak in uncorrected light curve; do not preform iterative
       outlier rejection
    2. No large peak, but no significant outliers; do not preform
       iterative outlier rejection
    3. No large peak; significant outliers; iterate

    """
    P1 = h5.attrs['P1_FFA']
    P2 = h5.attrs['P2_FFA']

    it           = 1
    done         = False
    pk_grass_fac = 4 
    maxit        = 5
    h5.attrs['itOutRej'] = True

    curIt = h5['it0']

    # Pull out the data columns from h5 file
    res0 = curIt['RES'][:]   
    lc0  = h5['mqcal'][:]
    fm   = ma.masked_array(lc0['fcal'],lc0['fmask'],fill_value=0)
    t    = lc0['t']
    cad  = lc0['cad']

    pks2n = max(res0['s2n'])

    # Determine whether outlier rejection is necessary
    Pcad  = res0['Pcad']
    bins  = np.logspace(np.log10(Pcad[0]),np.log10(Pcad[-1]+1),21)
    id    = np.digitize(Pcad,bins)
    p     = [np.percentile(res0['s2n'][id==i],90) for i in np.unique(id)]
    grass = np.rec.array(zip(p,bins[:-1]),names='p,bins')
    mgrass = np.median(p)
    print "Grass = %0.2f, Highest peak = %0.2f" %(mgrass,pks2n)
    if mgrass * pk_grass_fac < pks2n:
        print "Peak much larger than background.  Proceeding w/o itOutRej"
        h5.attrs['itStatus'] = 1 # Peak.  it0/RES, mqcal
        return
    
    np.random.seed(0)
    PcadG = np.random.random_integers(Pcad[0],Pcad[-1],size=50)

    # Determine the outlier level. Like the computation inside the
    # loop, we sample the periodogram at 50 points
    res = tdmarg(tdpep(t,fm,PcadG,twdG))
    cadCt = cadCount(cad,res)

    p90 = np.percentile(cadCt,90)
    p99 = np.percentile(cadCt,99)
    drop10x = p99-p90
    maCadCnt= p90+drop10x*4        
    print "p90 p99 outlier"
    print "%i  %i  %i  " % (p90,p99,maCadCnt)
    print "it nout fmask"

    # Iteratively remove cadences that are over represented in the
    # periodogram
    while done is False:
        # Store cadCt to current iteration
        curIt['cadCt'] = cadCt

        # Compute outliers and update the mask
        bout  = cadCt > maCadCnt
        nout  = cadCt[bout].size
        bout  = nd.convolve(bout.astype(float),np.ones(20) ) >  0
        fm.mask = fm.mask | bout
        print "%i  %i  %i" % (it,nout,fm.count() )

        # If there are no outliers on the first iteration, there is no
        # reason to Recompute the full grid. Set the top

        if (nout==0) and (it==1):
            done = True              
            h5.attrs['itStatus'] = 2 # No Peak.  it0/RES, mqcal
        elif (nout==0 ) or (it > maxit):
            done = True
        else:
            h5.attrs['itStatus'] = 3 # No Peak.  it0/RES, mqcal
            # Recalculate grid
            res   = tdmarg(tdpep(t,fm,PcadG,twdG))
            cadCt = cadCount(cad,res) 

            # Store away the current iteration
            curIt = h5.create_group('it%i' % it)
            curIt['RES']   = res
            curIt['fmask'] = fm.mask

        it +=1

    if it>1:
        print "Reruning grid full grid search with outliers gone"
        PcadG = np.arange(h5.attrs['P1_FFA'],h5.attrs['P2_FFA'])
        del curIt['RES']
        curIt['RES'] = tdmarg(tdpep(t,fm,PcadG,twdG))



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

def tdpep(t,fm,PcadG,twdG):
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


def tdpep2(t,fm,Pcad0,twdG,noiseG):
    """

    The cubes are 3D datasets with 
    sCube[i,j,k] 
    - i Specifies twdG
    - j Specifies period
    - k specifies epoch

    """
    assert fm.fill_value ==0

    # Determine the grid of periods that corresponds to integer
    # multiples of cadence values

    ntwd  = len(twdG)
    fmW = FFA.XWrap2(fm,Pcad0,pow2=True)
    mask = (~fmW.mask).astype(float)
    fmW.fill_value=0


    # Period grid
    M   = fmW.shape[0]  # number of rows
    idRow = np.arange(M,dtype=int)       # id of each row
    Pcad  = Pcad0 + idRow.astype(float) / (M - 1)

    # epoch grid
    idCol = np.arange(Pcad0,dtype=int)   # id of each column    

    XsumP  = FFA.FFA(fmW.filled()).astype(float)
    XcntP  = FFA.FFA(mask).astype(float)

#    resL = []
    return FBLS.cmaxDelTt0(XsumP,XcntP,Pcad0,twdG,noiseG,twdG.size)
#        resL.append(res)
        
#    names = 's2nMa,iMa,kMa'.split(',')
#    types = [float,int,int]
#    res = np.array(resL,dtype=zip(names,types)  )

#    return res


def noise(t,fm,twdG):
    # Constant estimation fof the noise for deltaT
    noiseG = []
    for twd in twdG:
        dM     = mtd(t,fm,twd)
        noiseG.append( ma.median( ma.abs(dM) )  )
    noiseG = np.array(noiseG)
    return noiseG

import config
import keptoy
from keptoy import P2a,a2tdur

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
    d = dict( qmi=qmi, qma=qma, fmi=fmi, nf=nf, df=df, farr=farr,nb=nb ,
              delT1=delTlim[0], delT2=delTlim[1], Pcad1=Pcad1, Pcad2=Pcad2 )
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

import time
import pandas

filtWid = np.array([5,10,20]) # lists at which to compute filters
def tryalgs(FdtL,W,alg,d):
    starttime=time.time()
    ver = True
    Wmi = filtWid[filtWid-d['delT1'] >= 0][0]
    Wma = filtWid[filtWid-d['delT2'] >= 0][0]
    delTarr = filtWid[(filtWid >= Wmi) & (filtWid <= Wma)]
    ndelT = len(delTarr)
    # run the search over different time scales.

    FOM = []
    for i in range(ndelT):
        if i==0:
            delT1 = d['delT1']
        else:
            delT1 = delTarr[i-1]
        delT2 = delTarr[i]
        # Load in the appropriate flux time series

        Fdt = FdtL['%i' % delT2]

        Pcad,out = FBLS(Fdt,W,alg,d['Pcad1'],d['Pcad2'],delT1,delT2)
        df = pandas.DataFrame(out)
        FOM.append(np.array(df[0]))

    Parr = Pcad*config.lc
    FOM = np.max(np.vstack(FOM),axis=0)

    stoptime = time.time()


    if ver:
        print "total execution time %.2f " % (stoptime-starttime)

    return Parr,FOM


def FBLS(F,W,alg,d):

    """
    algs can be FBLS_SNR, FBLS_SRCC, FBLS_SRpos
    
    """
    outL = []
    Pcad = np.array([])
    for Pcad0 in range(d['Pcad1'],d['Pcad2']+1):
        s0W = FFA.XWrap2(W,Pcad0,pow2=True)
        s0W.fill_value=0
        s1W = FFA.XWrap2(F,Pcad0,pow2=True)
        s1W.fill_value=0

        ss0 = FFA.FFA(s0W.filled()).astype(float)
        ss1 = FFA.FFA(s1W.filled()).astype(float)

        if alg=='FBLS_SNR':
            s2W = FFA.XWrap2(F**2,Pcad0,pow2=True)
            s2W.fill_value=0
            ss2 = FFA.FFA(s2W.filled()).astype(float)

        ss0 = FFA.FFA(s0W.filled()).astype(float)
        ss1 = FFA.FFA(s1W.filled()).astype(float)
        M     = s0W.shape[0]  # number of rows
        idRow = np.arange(M,dtype=int)       # id of each row
        Pcad  = np.append(Pcad, Pcad0 + idRow.astype(float) / (M - 1))
        for i in range(ss0.shape[0]):
            if alg=='FBLS_SRCC':
                out = FBLS_cy.FBLS_SRCC(ss1[i],ss0[i],
                                        d['qmi'],d['qma'],ss0[i].size)
            elif alg=='FBLS_SRpos':
                out = FBLS_cy.FBLS_SRpos(ss1[i],ss0[i],
                                         d['delT1'],d['delT2'],ss0[i].size)
            elif alg=='FBLS_SNR':
                out = FBLS_cy.FBLS_SNR(ss1[i],ss0[i],ss2[i],
                                       d['delT1'],d['delT2'],ss0[i].size)
            outL.append(out)

    return Pcad,outL

##### BLS-SNR ######

def BLS_SNR(t,fm,P1,P2):
    """
    BLS based on SNR
    """
    df = pgramParsSeg(P1,P2,t.ptp(),nseg=5)

    SNR  = []
    Parr = []
    for i in range(len(df)):
        d = df[i]
        SNR.append( multiScaleBLS(t,fm,d) )
        Parr.append( 1. / d['farr'] )

    SNR  = np.hstack(SNR)
    Parr = np.hstack(Parr)
    return Parr,SNR


from scipy import ndimage as nd
def medfilt_interp(fm,s):
    """
    Run a median filter on a array with masked values. 
    I'll interpolate between with missing values, not quite a true median filter
    """
    x  = np.arange(fm.size)
    xp = x[~fm.mask]
    fp = fm.compressed()

    fi   = np.interp(x,xp,fp)
    fmed = nd.median_filter(fi,size=s)
    return fmed


import BLS_cy
import BLS_cext

def multiScaleBLS(t0,fm,d,alg):
    fact = 3 # Filter is fact times wider than maximum duration searched over
    SNR2D = []
    for i in [0,1]:
        t = t0.copy()
        if i==0:
            delT1 = d['delT1']
            delT2 = d['delT2'] * 0.5
        else:
            delT1 = d['delT2'] * 0.5
            delT2 = d['delT2']
        
        qmi = float(delT1) / d['Pcad1']
        qma = float(delT2) / d['Pcad1']

        # preserve times up to delT[i+1]
        filtW = fact*delT2

        fmed = medfilt_interp(fm,filtW)

        print "%(P1).2f %(P2).2f " % d, filtW
        
        fbls = fm - fmed
        Parr = 1./d['farr']
        t    = t[~fm.mask]
        fbls = fbls.compressed()

        if alg=='cy':
            SNR = BLS_cy.BLS_SNR(t,fbls,Parr,d['nb'],qmi,qma)
        elif alg=='cext':
            SNR = BLS_cext.cBLS(t,fbls,Parr,d['nb'],qmi,qma)
        SNR2D.append(SNR)
    
    return np.vstack(SNR2D).max(axis=0)

def ebls(t,fm,P1,P2,alg,dv=False):
    dL = pgramParsSeg(P1,P2,t.ptp(),nseg=10)
    df = pandas.DataFrame(dL)
    
    SNR  = []
    Parr = np.hstack(df.farr)
    Parr = 1./Parr

    def core(d):
        SNR2D = multiScaleBLS(t,fm,d,alg)
        return np.vstack(SNR2D).max(axis=0)
    
    if dv!=False:
        dv.push(dict(t=t,fm=fm))
        f = lambda d : multiScaleBLS(t,fm,d,'cext')
        test = dv.map(f,dL,block=True)

        SNR = dv.map(core,dL,block=True)
    else:
        SNR = map(core,dL)

    SNR = np.hstack(SNR)
    return Parr,SNR



def pebls(t,fm,P1,P2,alg,dv=False):
    dL = pgramParsSeg(P1,P2,t.ptp(),nseg=10)
    df = pandas.DataFrame(dL)
    
    SNR  = []
    Parr = np.hstack(df.farr)
    Parr = 1./Parr

    def core(d):
        SNR2D = multiScaleBLS(t,fm,d,alg)
        return 
    
    if dv!=False:
        SNR = dv.map(core,dL,block=True)
    else:
        SNR = map(core,dL)

    SNR = np.hstack(SNR)
    return Parr,SNR

