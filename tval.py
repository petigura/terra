"""
Transit Validation

After the brute force period search yeilds candidate periods,
functions in this module will check for transit-like signature.
"""
import numpy as np
from numpy import ma

from scipy import optimize
import scipy.ndimage as nd

import glob
import copy
import keptoy
import tfind

def trsh(P,tbase):
    ftdurmi = 0.5
    tdur = keptoy.a2tdur( keptoy.P2a(P) ) 
    tdurmi = ftdurmi*tdur 
    dP     = tdurmi * P / tbase
    depoch = tdurmi

    return dict(dP=dP,depoch=depoch)

def objMT(p,time,fdt,p0,dp0):
    """
    Multitransit Objective Function

    With a prior on P and epoch

    """
    fmod = keptoy.P05(p,time)
    resid = (fmod - fdt)/1e-4
    obj = (resid**2).sum() + (((p0[0:2] - p[0:2])/dp0[0:2])**2 ).sum()
    return obj

def obj1T(p,t,f,P,p0,dp0):
    """
    Single Transit Objective Function
    """
    model = keptoy.P051T(p,t,P)
    resid  = (model - f)/1e-4
    obj = (resid**2).sum() + (((p0[0:2] - p[0:2])/dp0[0:2])**2 ).sum()
    return obj


def obj1Tlin(pNL,t,f):
    """
    Single Transit Objective Function.  For each trial value of epoch
    and width, we determine the best fit by linear fitting.

    Parameters
    ----------
    pNL  - The non-linear parameters [epoch,tdur]
   
    """
    pL = linfit1T(pNL,t,f)
    pFULL = np.hstack( (pNL[0],pL[0],pNL[1],pL[1:]) )
    model = keptoy.P051T(pFULL,t)

    resid  = ((model - f)/1e-4 )
    obj = (resid**2).sum() 
    return obj


def linfit1T(p,t,f):
    """
    Linear fit to 1 Transit.

    Depth and polynomial cofficents are linear

    Parameters
    ----------
    p : [epoch,tdur]
    t : time
    f : flux

    Returns
    -------
    p1 : Best fit [df,pleg0,pleg1...] from linear fitting.
    """
    
    ndeg=3

    epoch  = p[0]
    tdur   = p[1]

    # Construct polynomial design matrix
    trendDS = [] 
    for i in range(ndeg+1):
        pleg = np.zeros(ndeg+1)
        pleg[i] = 1
        trendDS.append( keptoy.trend(pleg,t) )
    trendDS = np.vstack(trendDS)

    # Construct lightcurve design matrix
    plc = np.hstack(( epoch,1.,tdur,list(np.zeros(ndeg+1)) ))
    lcDS = keptoy.P051T(plc,t)

    DS = np.vstack((lcDS,trendDS))
    p1 = np.linalg.lstsq(DS.T,f)[0]

    return p1

def fit1T(pNL0,t,f):
    """
    Fit Single transit
    """
    dpNL0 = np.array([0.2,0.2])

    objp = lambda p,t,f : obj1Tlin(p,t,f) + (((p-pNL0)/dpNL0)**2).sum()
    pNL = optimize.fmin(objp,pNL0,args=(t,f),disp=False)
    pL = linfit1T(pNL,t,f)
    pFULL = np.hstack( (pNL[0],pL[0],pNL[1],pL[1:]) )
    if pFULL[1] < 0:
        pFULL[1] = 0

    return pFULL

def LDT(t,fm,p,wd=2.):
    """
    Local detrending.  
    At each putative transit, fit a model transit and continuum lightcurve.

    Parameters
    ----------

    t  : Times (complete data string)
    fm : Flux. bad values masked out.
    p  : Parameters {'P': , 'epoch': , 'tdur': }

    """
    P     = p['P']
    epoch = p['epoch']
    tdur  = p['tdur']

    Pcad     = round(P/keptoy.lc)
    epochcad = round(epoch/keptoy.lc)
    tdurcad  = round(tdur/keptoy.lc)
    wdcad    = round(wd/keptoy.lc)

    dM = tfind.mtd(t,fm.filled(),tdurcad)
    dM.mask = fm.mask | ~tfind.isfilled(t,fm,tdurcad)

    tm   = ma.masked_array(t,copy=True,mask=fm.mask)
    ### Determine the indecies of the points to fit. ###
    # Exclude regions where the convolution returned a nan.
    ms   = midTransId(t,p)
    ms   = [m for m in ms if ~dM.mask[m] ]
    sLDT = [ getSlice(m,wdcad) for m in ms ]
    x = np.arange(dM.size)
    idL  = [ x[s][np.where(~fm[s].mask)] for s in sLDT ]

    func = lambda m,id : fit1T( [t[m],tdur], tm[id].data , fm[id].data) 
    p1L = map(func,ms,idL)

    return p1L,idL

def fitcand(t,fm,p0,ver=True):
    """
    Fit Candidate Transits

    Starting from the promising (P,epoch,tdur) combinations returned by the
    brute force search, perform a non-linear fit for the transit.

    Parameters
    ----------

    t      : Time series  
    fm     : Flux
    p0     : Dictionary {'P':Period,'epoch':Trial epoch,'tdur':Transit Duration}

    """
    twdcad = 2./keptoy.lc
    P     = p0['P']
    epoch = p0['epoch']
    tdur  = p0['tdur']

    p1L,idL = LDT(t,fm,p0)
    nT = len(p1L)
    tdt,fdt = dt1T(t,fm,p1L,idL)

    p0 = np.array([P,epoch,0.e-4,tdur])
    fitpass = False
    if (nT >= 3) :
        tbase = t.ptp()
        dp0 =  trsh(P,tbase)
        dp0 = [dp0['dP'],dp0['depoch']]
        p1  = optimize.fmin(objMT,p0,args=(tdt,fdt,p0,dp0) ,disp=False)
        tfold = tfind.getT(tdt,p1[0],p1[1],p1[3])
        fdt2 = ma.masked_array(fdt,mask=tfold.mask)
        if fdt2.count() > 20:
            s2n = - ma.mean(fdt2)/ma.std(fdt2)*np.sqrt( fdt2.count() )
            fitpass = True
        else: 
            fitpass = False
            s2n = 0
        if ver:
            print "%7.02f %7.02f %7.02f" % (p1[0] , p1[1] , s2n )

    # To combine tables everythin must be a float.
    if fitpass:
        res = dict( P=p1[0],epoch=p1[1],df=p1[2],tdur=p1[3],s2n=s2n )
        return res
    else:
        return dict( P=p0[0],epoch=p0[1],df=p0[2],tdur=p0[3],s2n=0. )


def dt1T(t,fm,p1L,idL):
    """
    Detrend based on single transit.
    """
    fdt = np.empty(t.size)
    for p1,id in zip(p1L,idL):
        fdt[id] = fm.data[id] - keptoy.trend(p1[3:],t[id])    

    id  = np.hstack(idL)
    tdt = t[id]
    fdt = fdt[id]

    return tdt,fdt


def fitcandW(t,fm,dL,view=None,ver=True):
    """
    """
    n = len(dL)

    return resL


def parGuess(res,nCheck=50):
    """
    Parameter guess

    Given the results of the matched filter approach, return the guess
    values for the non-linear fitter.

    Parameters
    ----------

    res - Dictionary with the following keys:

        s2n   : Array of s2n
        PG    : Period grid
        epoch : Array of epochs
        twd   : Array of epochs
    
    Optional Parameters
    -------------------

    nCheck : How many s2n points to look at?

    Notes
    -----

    Right now the transit duration is hardwired at 0.3 days.  This it
    should take the output value of the matched filter.

    """

    idCand = np.argsort(-res['s2n'])
    dL = []
    for i in range(nCheck):
        idx = idCand[i]
        d = dict(P=res['PG'][idx],epoch=res['epoch'][idx],
                 tdur=res['twd'][idx]*keptoy.lc)
        dL.append(d)

    return dL

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

    Pcad     = int(round(P/keptoy.lc))
    epochcad = int(round( (epoch-t[0])/keptoy.lc )  )

    nT = t.size/Pcad + 1  # maximum number of transits

    ### Determine the indecies of the points to fit. ###
    ms = np.arange(nT) * Pcad + epochcad
    ms = [m for m in  ms if (m < t.size) & (m > 0) ]
    return ms

def aliasW(t,f,resL0):
    """
    Alias Wrap

    """

    s2n = np.array([ r['s2n'] for r in resL0])
    assert ( s2n > 0).all(),"Cut out failed fits"

    resL = copy.deepcopy(resL0)

    for i in range(len(resL0)):
        X2,X2A,pA,fTransitA,mTransit,mTransitA = alias(t,f,resL0[i])
        if X2A < X2:
            res = fitcand(t,f,pA)
            resL[i] = res

    return resL

def alias(t,fm,p):
    """
    Evaluate the Bayes Ratio between signal with P and 2 *P

    Parameters
    ----------

    t : Time series
    f : Flux series
    p : Parameter dictionary.
    
    """

    pA = copy.deepcopy(p)
    pA['P'] = 0.5 * pA['P']
    
    p1L,idL = LDT(t,fm,pA)
    tdt,fdt = dt1T(t,fm,p1L,idL)           
    
    pl  = [p['P'],p['epoch'],p['df'],p['tdur']]
    plA = [pA['P'],pA['epoch'],pA['df'],pA['tdur']]

    model  = keptoy.P05(pl  , tdt )
    modelA = keptoy.P05(plA , tdt )

    tTransitA = tfind.getT(tdt,pA['P'],pA['epoch'],pA['tdur'])
    
    mTransit  = ma.masked_array(model,copy=True,mask=tTransitA.mask)
    mTransitA = ma.masked_array(modelA,copy=True,mask=tTransitA.mask)

    fTransitA = ma.masked_array(fdt,copy=True,mask=tTransitA.mask)

    X2  = ma.sum( (fTransitA - mTransit)**2 )
    X2A = ma.sum( (fTransitA - mTransitA)**2 )

    print "Input Period Chi2 = %e, Alias Chi2 = %e " % (X2, X2A)

    return X2,X2A,pA,fTransitA,mTransit,mTransitA
    

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









