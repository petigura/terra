"""
Transit Validation

After the brute force period search yeilds candidate periods,
functions in this module will check for transit-like signature.
"""
import numpy as np
from numpy import ma
from scipy import optimize

import copy
import keptoy
import tfind
from numpy.polynomial import Legendre

dP     = 0.5
depoch = 0.5

def val(t,fm,tres):
    """
    Validate promising transits.

    Parameters
    ----------
    t    : time
    fm   : flux 
    tres : a record array with the parameters of the putative transit.
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

    r2d = lambda r : dict(P=r['P'],epoch=r['epoch'],tdur=r['tdur'],df=r['df'])
    dL = map(r2d,tres)

    fcand = lambda d : fitcand(t,fm,d)
    resL = map(fcand,dL)

    # Cut out crazy fits.
    resL = [r for r in resL if r['stbl'] ]

    ### Alias Lodgic ###
    # Check the following periods for aliases.
    resL = [r for r in resL if  r['s2n'] > 5]


    falias = lambda r : aliasW(t,fm,r)
    resL = map(falias,resL)            

    dtype = [('P',float),('epoch',float),('tdur',float),('df',float),
             ('s2n',float)]
    
    d2l = lambda d : tuple([ d[ k[0] ] for k in dtype ])
    d2r = lambda d : np.array( d2l(d) ,dtype=dtype)

    rval = map(d2r,resL)
    rval = np.hstack(rval)

    return rval


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

    With a prior on P and epoch

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

    dM = tfind.mtd(t,fm.filled(),tdurcad)
    dM.mask = fm.mask | ~tfind.isfilled(t,fm,tdurcad)

    ### Determine the indecies of the points to fit. ###
    # Exclude regions where the convolution returned a nan.
    ms   = midTransId(t,p)
    if usemask:
        ms   = [m for m in ms if ~dM.mask[m] ]

    sLDT = [ getSlice(m,wdcad) for m in ms ]
    x = np.arange(dM.size)
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
    res : result dictionary.
    
    """
    dtL  = LDTwrap(t,fm,p)
    dt   = np.hstack(dtL)

    fdt = dt['fdt']
    tdt = dt['tdt']

    p0  = np.array([p['P'],p['epoch'],p['df'],p['tdur']])
    p1  = optimize.fmin_powell(objMT,p0,args=(tdt,fdt),disp=False)

    dp = (p0[:2]-p1[:2])
    if (abs(dp) > np.array([dP,depoch])).any():
        stbl = False
    elif (p1[0] < 0) | (p1[3] < 0):
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



def aliasW(t,fm,p0):
    """
    Alias Wrap
    """
    p = copy.deepcopy(p0)

    X2,X2A = alias(t,fm,p)
    if X2A < X2:
        p['P'] = p['P']*0.5
        return p
    else:
        return p

def alias(t,fm,p):
    """
    Evaluate the Bayes Ratio between signal with P and 0.5*P

    Parameters
    ----------

    t  : Time series
    fm : Flux series
    p  : Parameter dictionary.
    
    """
    pA = copy.deepcopy(p)
    pA['P'] = 0.5 * pA['P']
    resL = LDTwrap(t,fm,pA)
    res  = np.hstack(resL)

    # Masked array corresponding to P = 2 P
    tfold  = getT(res['tdt'],pA['P'],pA['epoch'],pA['tdur'])

    tT     = ma.masked_array(res['tdt'],copy=True,mask=tfold.mask)
    fT     = ma.masked_array(res['fdt'],copy=True,mask=tfold.mask)
    
    X2 = lambda par : ma.sum( (fT - keptoy.P05(pd2a(par),tT))**2 )
    return X2(p),X2(pA)


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
