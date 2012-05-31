"""
Transit Validation

After the brute force period search yeilds candidate periods,
functions in this module will check for transit-like signature.
"""
import numpy as np
from numpy import ma
from scipy import optimize
from scipy import ndimage as nd
import qalg

import copy
import keptoy
import tfind
from numpy.polynomial import Legendre
from matplotlib import mlab

from config import *

def val(t,fm,tres):
    """
    Validate promising transits.

    Wrapper function which does the following for a list of transit
    specified via `tres`:
    - Fits the transit with Protopapas (2005) `keptoy.P05` model.
    - For transits with S/N > hs2n look for harmonics.

    Parameters
    ----------
    t    : time
    fm   : masked flux array.
    tres : a record array with the parameters of the putative transit.
           - s2n. val evaluates transits with the highest S/N.
           - P
           - epoch
           - tdur
           - df
           
    Returns
    -------
    rval : a record array with following fields:
           - P     <- From best fit
           - epoch
           - tdur
           - df
           - s2n
    """


    trespks = tres[ nd.maximum_filter(tres['s2n'],3) == tres['s2n']  ]
    lcFitThresh = np.sort( trespks['s2n'] )[-nCheck]
    trespks = trespks[ trespks['s2n'] >  lcFitThresh ]

    def fcand(r):
        d = qalg.rec2d(r)
        resd = fitcand(t,fm,d)
        return qalg.d2rec(resd)

    tresfit = map(fcand,trespks)
    tresfit = np.hstack(tresfit)
    tresfit = tresfit[ tresfit['stbl'] ]   # Cut out crazy fits.

    def fharm(r):
        d = qalg.rec2d(r)
        resd = harmW(t,fm,d)
        return qalg.d2rec(resd)
    
    tresharm = map(fharm,tresfit[ tresfit['s2n'] > hs2n] )
    tresharm = np.hstack(tresharm)

    return tresfit,tresharm


def getT(time,P,epoch,wd):
    """
    Get Transits

    time : Time series
    P    : Period
    epoch: epoch
    wd   : How much data to return for each slice.

    Returns
    -------
    Time series phase folded with everything but regions of width `wd`
    centered around epoch + (0,1,2,...)*P

    Note
    ----

    plot(tfold,fm) will not plot all the transit for some reason.
    Instead, do plot(tfold[~tfold.mask],fm[~tfold.mask])

    """
    tfold = time - epoch + P /2 #  shift transit to (0.5, 1.5, 2.5) * P
    tfold = np.mod(tfold,P)     #  Now transits are at 0.5*P
    tfold -= P/2
    tfold = ma.masked_outside(tfold,-wd/2,wd/2)
    tfold = ma.masked_invalid(tfold)
    return tfold

def trsh(P,tbase):
    ftdurmi = 0.5
    tdur = keptoy.a2tdur( keptoy.P2a(P) ) 
    tdurmi = ftdurmi*tdur 
    dP     = tdurmi * P / tbase
    depoch = tdurmi

    return dict(dP=dP,depoch=depoch)
    
def objMT(p,time,fdt):
    """
    Multitransit Objective Function
    """
    fmod = keptoy.P05(p,time)
    resid = (fmod - fdt)/1e-4
    obj = (resid**2).sum() 
    return obj

def obj1T(p,t,f):
    """
    Single Transit Objective Function
    """
    model = keptoy.P051T(p,t)
    resid  = (model - f)/1e-4
    obj = (resid**2).sum()
    return obj


def id1T(t,fm,p,wd=2.,usemask=True):
    """
    Grab the indecies and midpoint of a putative transit. 
    """
    tdur  = p['tdur']

    tdurcad  = round(tdur/keptoy.lc)
    wdcad    = round(wd/keptoy.lc)

    mask = fm.mask | ~tfind.isfilled(t,fm,tdurcad)

    ### Determine the indecies of the points to fit. ###
    # Exclude regions where the convolution returned a nan.
    ms   = midTransId(t,p)
    if usemask:
        ms   = [m for m in ms if ~mask[m] ]

    sLDT = [ getSlice(m,wdcad) for m in ms ]
    x = np.arange(fm.size)
    idL  = [ x[s][~fm[s].mask] for s in sLDT ]
    idL  = [ id for id in idL if id.size > 10 ]

    return ms,idL

def LDT(epoch,tdur,t,f,pad=0.2,deg=1):
    """
    Local detrending

    A simple function that subtracts a polynomial trend from the
    lightcurve excluding a region around the transit.

    pad : Extra number of days to notch out of the of the transit region.
    """
    bcont = abs(t - epoch) > tdur/2 + pad
    fcont = f[bcont]
    tcont = t[bcont]

    legtrend = Legendre.fit(tcont,fcont,deg,domain=[t.min(),t.max()])
    trend    = legtrend(t)
    return trend 

def LDTwrap(t,fm,p):
    ms,idL = id1T(t,fm,p,wd=2.)
    tdur  = p['tdur']

    dtype=[('trend',float),('fdt',float),('tdt',float) ]
    resL = []
    for m,id in zip(ms,idL):
        trend = LDT( t[m],tdur, t[id] , fm[id].data)
        fdt   = fm[id]-trend
        tdt   = t[id]
        res = np.array(zip(trend,fdt,tdt),dtype=dtype)
        resL.append(res)
    return resL

def fitcand(t,fm,p,full=False):
    """
    Perform a non-linear fit to a putative transit.

    Parameters
    ----------
    t  : time
    fm : flux
    p  : trial parameter (dictionary)
    full : Retrun tdt and fdt

    Returns
    -------
    res : Dictionary with the following parameters.
          - P
          - epoch
          - df
          - tdur
          - s2n
          - stbl    = boolean variable if fit is well behaved.
          - tdt  - time
          - fdt  - detrended flux


    Todo
    ----

    Add priors in for P.  tdur goes negative, but the sign doesn't matter
    """
    dtL  = LDTwrap(t,fm,p)
    dt   = np.hstack(dtL)

    fdt = dt['fdt']
    tdt = dt['tdt']

    p0  = np.array([p['P'],p['epoch'],p['df'],p['tdur']])
    p1  = optimize.fmin_powell(objMT,p0,args=(tdt,fdt),disp=False)

    # Hack to prevent negative transit duration.
    p1[3] = np.abs(p1[3])

    dp = (p0[:2]-p1[:2])
    if (abs(dp) > np.array([dP,depoch])).any():
        stbl = False
    elif p1[0] < 0:
        stbl = False
    else:
        stbl = True

    tfold = getT(tdt,p['P'],p['epoch'],p['tdur'])
    fdt   = ma.masked_array(fdt,mask=tfold.mask)
    tdt   = ma.masked_array(tdt,mask=tfold.mask)

    s2n = s2n_fit(fdt,tdt,p1)
    res = dict(P=p1[0],epoch=p1[1],df=p1[2],tdur=p1[3],s2n=s2n,stbl=stbl)

    if full:
        res['fdt'] = fdt
        res['tdt'] = tdt

    return res

def s2n_mean(fdt):
    return -ma.mean(fdt)/ma.std(fdt)*np.sqrt(fdt.count())

def s2n_med(fdt):
    sig = ma.median(fdt)
    noise = ma.median(abs(fdt-sig))
    return -sig/noise*np.sqrt(fdt.count())

def s2n_fit(fdt,tdt,p):
    """
    Evaluate S/N taking the best fit depth as signal and the scatter
    about the residuals as the noise.
    """
    model = keptoy.P05(p,tdt)
    sig   = p[2]
    resid = fdt-model
    noise = ma.median(abs(resid))
    s2n = sig/noise*np.sqrt(fdt.count() )
    return s2n

thresh = 0.001

def window(fl,PcadG):
    """
    Compute the window function.

    The fraction of epochs that pass our criteria for transit.
    """

    winL = []
    for Pcad in PcadG:
        flW = tfind.XWrap(fl,Pcad,fill_value=False)
        win = (flW.sum(axis=0) >= 3).astype(float)
        npass = np.where(win)[0].size
        win =  float(npass) / win.size
        winL.append(win)

    return winL

def midTransId(t,p):
    """
    Mid Transit Index

    Return the indecies of mid transit for input parameters.

    Parameters
    ----------

    t - timeseries
    p - dictionary with 'P','epoch','tdur'

    """
    P     = p['P']
    epoch = p['epoch']

    epoch = np.mod(epoch,P)
    tfold = np.mod(t,P)
    
    ms = np.where(np.abs(tfold-epoch) < 0.5 * keptoy.lc)[0]
    ms = list(ms)
    return ms



def harmW(t,fm,p0,disp=False):
    """
    Harmonics

    The difference in MES between the true period and harmonics ( 1/2,
    2/3, .. )*P and subharmonics (2, 3/2, 4/3, ...) * P can be small.
    We compute Chi2 between the data and the modes and compare them
    for different models.  We select the mode with the lowest Chi2.
    This amounts to Bayesian model comparison.

    Evaluate the Bayes Ratio between signal with P and 0.5*P

    Parameters
    ----------

    t  : Time series
    fm : Flux series
    p0 : Parameter dictionary.
    

    Returns
    -------
    p  : The parameter dictionary with the best X2.

    """
    
    func = lambda h : harm(t,fm,p0,h,full_output=False)
    res = map(func,harmL)
    res = np.array(res, dtype = [('X20',float) , ('X2h',float)] )
    res = mlab.rec_append_fields(res,'h',harmL)
    res = mlab.rec_append_fields(res,'P',p0['P']*harmL)
    
    if disp:
        print mlab.rec2txt(res) 

    dX2 = res['X2h'] - res['X20'] # Difference in the Chi2
    if (dX2 > 0).all():
        return p0
    else:
        idmin = dX2.argmin()
        p = copy.deepcopy(p0)
        p['P'] = p0['P'] * res['h'][idmin]
        return p

def harm(t,fm,p0,h,full_output=False):
    """
    Harmonics

    Evaluate the difference in Chi2 for transit with P = p0['P'] and h*P.

    Parameters
    
    The difference in MES between the true period and harmonics ( 1/2,
    2/3, .. )*P and subharmonics (2, 3/2, 4/3, ...) * P can be small.
    We compute Chi2 between the data and the modes and compare them
    for different models.  We select the mode with the lowest Chi2.
    This amounts to Bayesian model comparison.

    Evaluate the Bayes Ratio between signal with P and 0.5*P

    Parameters
    ----------

    t  : Time series
    fm : Flux series
    p0 : Initial Parameter dictionary.
    h  : Harmonic period is p0['P']*h
    
    Returns
    -------
    
    X20    : Chi2 of starting model
    X2h    : Chi2 of harmonic model
    tTL    : Masked time used to compare models (full_output=True)
    fTL    : Masked flux used to compare models
    mod0   : Reference model
    modh   : Harmonic model
    """
    
    ph = copy.deepcopy(p0)
    ph['P'] = h * p0['P']

    # Masked array corresponding to the harmonic
    tfold0  = getT(t,p0['P'],p0['epoch'],p0['tdur'])
    tfoldh  = getT(t,ph['P'],ph['epoch'],ph['tdur'])

    # Only mask out points that don't belong in either the reference
    # or the harmonic photometry.
    mask    = (tfold0.mask & tfoldh.mask)
    tcomp   = ma.masked_array(t , mask,copy=True)
    fcomp   = ma.masked_array(fm, mask,copy=True)

    resL0 = LDTwrap(t,fm,p0)
    resLh = LDTwrap(t,fm,ph)

    resL = resL0 +resLh

    for r in resL:
        n   = r.size
        id0 = np.where(t == r['tdt'][0] )[0] # Find starting point
        fcomp[id0:id0+n] = r['fdt']

    fcomp.mask = mask

    def Chi2(p):
        mod = keptoy.P05( pd2a(p) , tcomp )
        res   = (fcomp - mod)/1e-4
        X2    = ma.sum( abs(res) )
        return mod,X2

    mod0,X20 = Chi2(p0)
    modh,X2h = Chi2(ph)

    if resL == []:
        return 
    elif len(resL) < 3:  # There must be 3 transits in the model
        return None 


    # If the harmonic has fewer than 3 transits, don't bother computing X2
    nh = len(resLh)
    if nh < 3:
        X2h = 1e3

    if full_output:
        return X20,X2h,tcomp,fcomp,mod0,modh
    else:
        return X20,X2h
    


def pd2a(d):
    return np.array([d['P'],d['epoch'],d['df'],d['tdur']])

def pa2d(a):
    return dict(P=a[0],epoch=a[1],df=a[2],tdur=a[3])

def getSlice(m,wdcad):
    """
    Get slice
    
    Parameters
    ----------
    m    : middle index (center of the slice).
    wdcad : width of slice list (units of cadence).

    """

    return slice( m-wdcad/2 , m+wdcad/2 )

def tdict(d,prefix=''):
    """
    
    """
    outcol = ['P','epoch','df','tdur']

    incol = [prefix+oc for oc in outcol]

    outd = {}
    for o,c in zip(outcol,incol):
        try:
            outd[o] = d[c]
        except KeyError:
            pass

    return outd

def nT(t,mask,p):
    """
    Simple helper function.  Given the transit ephemeris, how many
    transit do I expect in my data?
    """
    
    trng = np.floor( 
        np.array([p['epoch'] - t[0],
                  t[-1] - p['epoch']]
                 ) / p['P']).astype(int)

    nt = np.arange(trng[0],trng[1]+1)
    tbool = np.zeros(nt.size).astype(bool)
    for i in range(nt.size):
        it = nt[i]
        tt = p['epoch'] + it*p['P']
        # Find closest time.
        dt = abs(t - tt)
        imin = np.argmin(dt)
        if (dt[imin] < 0.3) & ~mask[imin]:
            tbool[i] = True

    return tbool
