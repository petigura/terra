import numpy as np
from numpy import ma
from scipy import ndimage as nd
from scipy.interpolate import UnivariateSpline,LSQUnivariateSpline
from scipy.optimize import fmin 
import os

import keptoy


dmi = 1.    # remove strips of data that are less than 1 day long.
gmi = 1/24. # Let's not worry about gaps that are 1 hour long
nq = 8
kepdir = os.environ['KEPDIR']
kepdat = os.environ['KEPDAT']


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

def nanIntrp(x0,y0,nContig=3):
    """
    Use linear interpolation to fill in the nan-points.

    nContig - Don't fill in nan values if there are more than a
    certain number of contiguous ones.
    """

    x,y = x0.copy(),y0.copy()

    y = ma.masked_invalid(y)
    x = ma.masked_array(x,mask=y.mask)
    xc,yc = x.compressed(),y.compressed()

    sp = UnivariateSpline(xc,yc,k=1,s=0)

    x.mask = ~x.mask
    sL = ma.notmasked_contiguous(x)
    for s in sL:
        nNans = s.stop-s.start
        if nNans <= nContig:
            y[s] = sp(x[s])

    return x.data,y.data

def segfitm(t,fm,bv):
    """
    Segment fit masked
    
    Parameters
    ----------
    t   : time 
    fm  : flux for a particular segment
    bv  : vstack of basis vectors
    
    """
    ncbv  = bv.shape[0]
    
    tm = ma.masked_array(t,copy=True,mask=fm.mask)
    mask  = fm.mask 

    bv = ma.masked_array(bv)
    bv.mask = np.tile(mask, (ncbv,1) )

    # Eliminate masked elements
    tseg = tm.compressed() 
    fseg = fm.compressed()
    bvseg = bv.compressed() 
    bvseg = bvseg.reshape(ncbv,bvseg.size/ncbv)

    fdtseg,ffitseg,p1seg = segfit(tseg,fseg,bvseg)

    return fdtseg,ffitseg

def segfit(tseg,fseg,bvseg):
    """
    Segment fit.

    Fit a segment of data with CBVs.

    Parameters
    ----------
    tseg  : time segment
    fseg  : flux segment
    bvseg : basis vector segments

    """
    ftnd  = spldt(tseg,fseg)
    bvtnd = [spldt(tseg,bvseg[i,:]) for i in range(bvseg.shape[0]) ] 
    bvtnd = np.vstack(bvtnd)

    bvdt = bvseg - bvtnd
    fdt = fseg - ftnd
    p1 = np.linalg.lstsq(bvdt.T,fdt)[0]

    ffit = modelCBV(p1,bvdt)
    return fdt,ffit,p1

def cbvseg(tm,segsep=1):
    """
    CBV Segmentation

    Split time series into segments for CBV fitting.  Sections are
    considered distinct if they are seperated by more than segsep days

    """
    nsep = segsep / keptoy.lc

    gap = ma.masked_array(tm.data,mask=~tm.mask,copy=True)
    nsep = segsep / keptoy.lc
    gapSlice = ma.notmasked_contiguous(gap)
    for g in gapSlice:
        if g.stop-g.start < nsep:
            tm.mask[g] = False

    return ma.notmasked_contiguous(tm)

def spldt(t,f,lendt=10):
    """
    Spline detrending
    """

    fm = ma.masked_invalid(f)
    assert fm.count() == fm.size,"No nans allowed."

    tbase = t.ptp()
    nknots = np.floor(tbase / lendt) + 1
    tk = np.linspace(t[0],t[-1],nknots)
    tk = tk[1:-1]
    sp = LSQUnivariateSpline(t,f,tk)
    return sp(t)

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
   

