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

# Load up multiquarter data.

cad = 30./60./24.
dmi = 1.    # remove strips of data that are less than 1 day long.
gmi = 1/24. # Let's not worry about gaps that are 1 hour long
nq = 8
kepdir = os.environ['KEPDIR']
cbvdir = os.path.join(kepdir,'kepdat/CBV/')

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

from scipy.interpolate import interp1d


def cbvDT(KIC):
    from pyraf import iraf
    iraf.kepler(_doprint=0)

    vectors = '1 2 3 4 5 6 7 8 9 10'
    method = 'llsq'
    fitpower = 2
    infiles = KICPath(KIC,'tempfits/clip/')

    tset = mqload(infiles)
    quarters = tset.tables.keys
    otset = atpy.TableSet()
    for q in quarters:

        cbvfile = glob.glob(kepdir+'kepdat/CBV/*q0%s*' % q[-1]) [0]
        iraf.kepler.kepcotrend(infile=tset[q].keywords['PATH'],
                               outfile='temp.fits',
                               clobber=True,
                               cbvfile=cbvfile,
                               vectors=vectors,
                               method=method,
                               fitpower=fitpower,
                               iterate=True,
                               sigmaclip='2',
                               maskfile='ranges/cut_bjd.txt',
                               scinterp='nearest',
                               plot=True
                               )

        infostr=  """
KIC     %i
Quarter %s
CBV %s
fitpower %.2f
method  %s 
""" % (KIC,q[-1],vectors.split()[-1],fitpower,method)

        
        t = atpy.Table('temp.fits',type='fits')
        t.table_name = q

        otset.append(t)
    otset.write(kepdir+'kepdat/DT/%s.fits' % KIC,type='fits',overwrite=True)
    


def larr(iL):
    oL = array([])
    for l in iL:
        oL = append(oL,l)
        
    return oL


def mqclip(t,f):
    rfile= os.path.join(kepdir,'ranges/cut_time.txt')
    t = ma.masked_array(t)
    f = ma.masked_array(f)

    rec = atpy.Table(rfile,type='ascii').data
    for r in rec:
        tt = ma.masked_inside( t,r['start'],r['stop'] )
        f.mask = f.mask | tt.mask
        
    f.fill_value = nan
    return f.filled()


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



def cbv(tQLC,fcol,efcol,cadmask=None,dt=False):
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

    Returns
    -------

    ffit    : The CBV fit to the fluxes.
    """
    cbv = [1,2,3]
    ncbv = len(cbv)
    deg = 5

    kw = tQLC.keywords
    assert kw['nQ'],'Assumes lightcurve has been normalized.' 

    # Load up the BV
    bvfile = os.path.join( cbvdir,'kplr*-q%02d-*.fits' % kw['QUARTER'])
    bvfile = glob.glob(bvfile)[0]
    bvhdu  = pyfits.open(bvfile)
    bvkw   = bvhdu[0].header
    bvcolname = 'MODOUT_%i_%i' % (kw['MODULE'],kw['OUTPUT'])
    tBV    = atpy.Table(bvfile,hdu=bvcolname,type='fits')
    assert bvkw['QUARTER'] == kw['QUARTER']," BV must be from the same quater"

    cad   = tQLC['CADENCENO' ]
    t     = tQLC['TIME'      ]
    f     = tQLC[fcol        ]
    ferr  = tQLC[efcol       ]

    tm = ma.masked_invalid(t)
    fm = ma.masked_invalid(f)
    mask  = tm.mask | fm.mask 

    if cadmask is not None:
        assert cadmask.size == cad.size, "Arrays must be of equal sizes."
        mask  = mask | cadmask # Add the time cut to the mask

    gid   = where(~mask)[0]
    bid   = where(mask)[0]

    t     = t[gid]
    f     = f[gid]
    cad   = cad[gid] 
    ferr  = ferr[gid] 

    # Construct a matrix of BV
    mbv = lambda i:  tBV['VECTOR_%i' % i][gid]
    bvectors = map(mbv,cbv)
    bvectors = vstack(bvectors)

    if dt:
        for i in range(ncbv):
            bvtrend = Legendre.fit(t,bvectors[i],deg)
            bvectors[i] = bvectors[i] - bvtrend(t)

        ftrend = Legendre.fit(t,f,deg)
        f = f - ftrend(t)

    data      = fm.data.copy()
    data[gid] = f.astype(float32)
    data[bid] = np.nan

    p0 = np.zeros( len(cbv) ) # Guess for parameters.


    p1,fopt ,iter ,funcalls, warnflag  = \
        fmin(objCBV,p0,args=(f,ferr,bvectors),disp=True,maxfun=10000,
             maxiter=10000,full_output=True,)

    if warnflag != 0:
        p1 = p0

    ffit      = fm.data.copy()
    ffit[gid] = modelCBV(p1,bvectors).astype(float32)
    ffit[bid] = np.nan    

    return data,ffit,bvectors,p1


def objCBV(p,f,ferr,bvectors):
   """
   Objective function for CBV vectors.
   """
   nres = ( f - modelCBV(p,bvectors) ) /ferr
   return np.sum( abs(nres) )

def modelCBV(p,bvectors):
   return np.dot(p,bvectors)
   
