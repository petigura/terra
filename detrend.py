from numpy import *
import glob
import pyfits
import atpy
import sys
from scipy import weave
from scipy.weave import converters
import os
from numpy.polynomial import Legendre
from keptoy import * 
from scipy import ndimage as nd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fmin 

cad = 30./60./24.
dmi = 1.    # remove strips of data that are less than 1 day long.
gmi = 1/24. # Let's not worry about gaps that are 1 hour long
nq = 8
kepdir = os.environ['KEPDIR']
kepdat = os.environ['KEPDAT']
cbvdir = os.path.join(kepdat,'CBV/')

def stitch(fl,tl,swd = 0.5):
    """
    Stitch together the boundaries of the quarters.
    """
    swd =  0.5 # How long a timeseries on either end to use days 
    nQ = len(fl)

    for i in range(nQ-1):
        fleft,tleft = fl[i],tl[i]
        fright,tright = fl[i+1],tl[i+1]

        lid = where(tleft > tleft[-1] - swd)[0]
        rid = where(tright < tright[0] + swd)[0]

        medleft = median(fleft[lid])
        medright = median(fright[rid])
        
        factor = medleft / medright
        fright *= factor

    return fl,tl

def larr(iL):
    oL = array([])
    for l in iL:
        oL = append(oL,l)
        
    return oL

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

def cbv(tQLC,fcol,efcol,cadmask=None,dt=False,ver=True):
    """
    Cotrending basis vectors.

    My implimentation of CBV detrending.  Assumes the relavent
    lightcurve has been detrended.

    Paramaters
    ----------

    tQLC    : Table for single quarter.
    fcol    : string.  name of the flux column
    efol    : string.  name of the flux_err colunm
    cadmask : Boolean array specifying a subregion
    ver     : Verbose output (turn off for batch).

    Returns
    -------

    ffit    : The CBV fit to the fluxes.
    """
    cbv = [1,2,3,4,5,6] # Which CBVs to use.
    ncbv = len(cbv)

    kw = tQLC.keywords
    assert kw['NQ'],'Assumes lightcurve has been normalized.' 

    cad   = tQLC['CADENCENO' ]
    t     = tQLC['TIME'      ]
    f     = tQLC[fcol        ]
    ferr  = tQLC[efcol       ]

    tm = ma.masked_invalid(t)
    fm = ma.masked_invalid(f)
    mask  = tm.mask | fm.mask 
    tm.mask = fm.mask = mask

    tBV = bvload(tQLC)
    bv = vstack( [tBV['VECTOR_%i' % i] for i in cbv] )
    bv = ma.masked_array(bv)
    bv.mask = np.tile(mask, (ncbv,1) )
    p1v = np.zeros(bv.shape)

    sL = ma.notmasked_contiguous(tm)
    sL = [s for s in sL if s.stop-s.start > 5 / lc]

    ffit  = np.zeros(fm.shape)
    fdtm  = fm.copy()
    

    for s in sL:
        fdt = lsdt(tm[s].data,fm[s].data)
        bvdt = [lsdt(tm[s].data,bv[i,s].data) for i in range(ncbv) ]
        bvdt = vstack(bvdt)
        p1 = np.linalg.lstsq(bvdt.T,fdt)[0]
        ffit[s] = modelCBV(p1,bvdt)
        fdtm[s] = fdt
        p1v[:,s] = np.tile( p1,(s.stop-s.start,1) ).T


    return fdtm,ffit,p1v

def lsdt(t,f):
    """
    Long/short timescale detrending.
    """

    tlong  = 10 # Cut out timescales longer than tlong days.
    tshort = 2./24 # Cut out timescales shorter than tshort days. 
    
    # Require that data is contiguous.
    assert max(t[1:]-t[0:-1]) < 1.2*lc

    sp = UnivariateSpline(t,f)
    fdt = f - sp(t)

    wid = tshort / lc
    fdt = nd.uniform_filter(fdt,size=4)
    return fdt
    
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
   

def bvload(tQLC):
    """
    Load basis vector.

    """    
    kw = tQLC.keywords
    assert kw['NQ'],'Assumes lightcurve has been normalized.' 

    bvfile = os.path.join( cbvdir,'kplr*-q%02d-*.fits' % kw['QUARTER'])
    bvfile = glob.glob(bvfile)[0]
    bvhdu  = pyfits.open(bvfile)
    bvkw   = bvhdu[0].header
    bvcolname = 'MODOUT_%i_%i' % (kw['MODULE'],kw['OUTPUT'])
    tBV    = atpy.Table(bvfile,hdu=bvcolname,type='fits')
    assert bvkw['QUARTER'] == kw['QUARTER']," BV must be from the same quater"
    return tBV
