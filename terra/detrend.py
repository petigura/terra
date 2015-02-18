import numpy as np
from numpy import ma
from scipy import ndimage as nd
from scipy.interpolate import UnivariateSpline,LSQUnivariateSpline
from scipy.optimize import fmin 
import os

import keptoy
import copy

dmi = 1.    # remove strips of data that are less than 1 day long.
gmi = 1/24. # Let's not worry about gaps that are 1 hour long
nq = 8

def GPdt(xi,x,y,corrlen=5):
    """
    Gaussian Process-based detrending

    Same sigature as interp(xi,x,y).
    """

    def kernel(a, b):
        """
        GP squared exponential kernel
        """
        sqdist  = np.sum(a**2,1).reshape(-1,1) + \
                  np.sum(b**2,1) - \
                  2*np.dot(a, b.T)

        return np.exp( -.5 * sqdist / corrlen**2 )

    X  = x[:,np.newaxis]
    Xi = xi[:,np.newaxis]

    K  = kernel(X,X)
    s  = 0.005    # noise variance.
    N  = len(X)   # number of training points.
    L  = np.linalg.cholesky(K + s*np.eye(N))
    Lk = np.linalg.solve(L, kernel(X, Xi) )
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    return mu

def bin(lc):
    """
    Bin the light curve for faster computation of GP

    Compute the mean of every nbin measurements (padding the end if
    necessary). Return only the valid datapoints.
    """

    fm = ma.masked_invalid( lc['f'] )
    nbin = 8
    rem  = np.remainder(lc.size,nbin)
    if rem > 0: # if points don't d
        npad = nbin - rem
        pad  = ma.masked_array(np.zeros(npad),True)
        fm = ma.hstack([fm,pad])

    y   = fm.reshape(-1,nbin).mean(axis=1)
    x   = lc['t'][::nbin]
    b   = ~y.mask
    return x[b],y.data[b]



def stitch(fl,tl,swd = 0.5):
    """
    Stitch together the boundaries of the quarters.
    """
    swd =  0.5 # How long a timeseries on either end to use days 
    nQ = len(fl)

    for i in range(nQ-1):
        fleft,tleft = fl[i],tl[i]
        fright,tright = fl[i+1],tl[i+1]

        lid = np.where(tleft > tleft[-1] - swd)[0]
        rid = np.where(tright < tright[0] + swd)[0]

        medleft = median(fleft[lid])
        medright = median(fright[rid])
        
        factor = medleft / medright
        fright *= factor

    return fl,tl

def maskIntrp(x0,y0,nContig=None):
    """
    Use linear interpolation to fill in masked pointss

    Parameters
    ----------
    x0      : Independent var
    y0      : Masked array.
    nContig : Skip if masked region contains more than a certain
              number of Contiguous masked points.  nContig = None means
              interpolate through everything.
    """

    x,y = x0.copy(),y0.copy()

    x = ma.masked_array(x,mask=y.mask,copy=True)
    xc,yc = x.compressed(),y.compressed()

    sp = UnivariateSpline(xc,yc,k=1,s=0)

    x.mask = ~x.mask
    sL = ma.notmasked_contiguous(x)

    if sL is None:
        return x.data,y

    if nContig is None:
        for s in sL:
            y[s] = sp(x[s])
    else:
        for s in sL:
            nNans = s.stop-s.start
            if nNans <= nContig:
                y[s] = sp(x[s])

    return x.data,y

def dt(t0):

    t = copy.deepcopy(t0)
    fm = ma.masked_array(t.f,mask=t.fmask)
    tm = ma.masked_array(t.TIME,mask=t.fmask)

    label = sepseg(tm)

    sL = ma.notmasked_contiguous(label)

    # If there is only one slice.
    if type(sL) == slice: 
        sL = [sL]

    id = sL2id(sL)

    tnd = fm.copy()
    temp = [spldtm(tm[s],fm[s]) for s in sL]
    temp = ma.hstack(temp)
    tnd[id] = temp
    return fm-tnd

def sepseg(tm0,tsep=1):
    """
    Seperate Segments

    Assign a label to segments.  Segments can have gaps no longer than
    tsep days.

    Parameters
    ----------

    tm   : masked array with time.  Gaps are labeled with mask.
    tsep : Segments can have gaps no longer than tsep.

    Returns
    -------
    
    label : Masked array.  Label identifies which segment a given point 
            belongs to.  Masked elements do not belong to any segment.

    Useage
    ------
    >>> label = sepseg(tm)
    >>> sL    = ma.notmasked_contiguous(label)
    `sL` is the list of slices corresponding to unique labels.
    """

    # Copy to prevent problems from changing the mask in place
    tm = tm0.copy() 
    label = np.zeros(tm.size)
    label[:] = np.nan

    nsep = tsep / keptoy.lc

    gap = ma.masked_array(tm.data,mask=~tm.mask,copy=True)
    nsep = tsep / keptoy.lc
    gapSlice = ma.notmasked_contiguous(gap)

    if gapSlice is None:
        label[:] = 0 
        return label
        
    for g in gapSlice:
        if g.stop-g.start < nsep:
            tm.mask[g] = False

    sL = ma.notmasked_contiguous(tm)
    nseg = len(sL)

    for i in range(nseg):
        s = sL[i]
        label[s] = i

    label = ma.masked_invalid(label)
    return label
    
def joinseg(lab1,lab2):
    """
    Join segments

    Create a new segment list from two.
    
    Parameters
    ----------
    
    lab1 : Labels define as per sepseg
    """
    tup0 = (lab1[~labj.mask][0],lab2[~labj.mask][0])
    j = 0
    for i in range(labj.size):
        if ~labj.mask[i]:
            tup = (lab1[i],lab2[i])
            if tup != tup0:
                j = j + 1
                tup0 = tup
            labj[i] = j

    return labj

def sL2id(sL):
    """
    Convert a slice list to indecies.
    """
    return np.hstack([np.mgrid[s] for s in sL])

def spldtm(t,fm,lendt=10):
    """
    Spline masked detrending.

    Compute a smooth detrend ignoring the masked points.


    Parameters
    ----------
    t     : ind var
    fm    : dep var (can be masked array)
    lendt : see spldt

    Returns
    -------
    tnd   : Best fit trend with the same mask as fm.
    
    """
    tm  = ma.masked_array(t,mask=fm.mask)
    tnd = ma.masked_array(fm,copy=True)
    x,y = tm.compressed(),fm.compressed()
    tnd[~fm.mask] = spldt(x,y,lendt=lendt).astype('float32')
    return tnd 

def spldt(t,f,lendt=10):
    """
    Spline detrending.

    The number of knots are chosen such that timescales longer than
    lendth are removed.

    Parameters
    ----------
    
    t     : independent variable
    f     : depedent variable
    lendt : length of timescale to remove.
    """

    assert f[np.isnan(f)].size == 0, "nans screw up spline."

    tbase = t.ptp()
    nknots = np.floor(tbase / lendt) + 1
    tk = np.linspace(t[0],t[-1],nknots)
    tk = tk[1:-1]
    sp = LSQUnivariateSpline(t,f,tk)
    tnd = sp(t)
    return tnd

def bvfit(t,f,ferr,bv):
    """
    Fit basis vectors:
    """
    
    p0 = np.ones( bv.shape[0] ) # Guess for parameters.

    p1,fopt ,iter ,funcalls, warnflag  = \
        fmin(objCBV,p0,args=(f,ferr,bv),disp=False,maxfun=10000,
             maxiter=10000,full_output=True)
    ffit = modelCBV(p1,bv).astype(float32)
    return p1,ffit

def objCBV(p,f,ferr,bv):
   """
   Objective function for CBV vectors.
   """
   nres = ( f - modelCBV(p,bv) ) /ferr
   return np.sum( abs(nres) )

def modelCBV(p,bv):
   return np.dot(p,bv)


def mmedian_filter(F,W,s):
    """
    Run a median filter on a array with masked values. 
    I'll interpolate between with missing values, not quite a true median filter
    """
    x  = np.arange(F.size)
    xp = x[W > 0]
    fp = F[W > 0]
    Fi = np.interp(x,xp,fp)
    return nd.median_filter(Fi,size=s)   

