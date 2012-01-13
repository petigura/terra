"""
Functions for facilitating the reading and writing of Kepler files.
"""
import numpy as np
from numpy import ma
from scipy.interpolate import UnivariateSpline
from matplotlib import mlab
from scipy import ndimage as nd
import copy
import atpy
import os
import glob
import pyfits

import keptoy
import detrend
import tfind


kepdir = os.environ['KEPDIR']

def KICPath(KIC,basedir):
    """
    KIC     - Target star identifier.
    basedir - Directory leading to files.
              'orig' - prestine kepler data
              'clip' - clipped data
              'dt'   - detrended data.
    """

    if basedir is 'orig':
        basedir = 'kepdat/EX/Q*/'
    if basedir is 'clip':
        basedir = 'tempfits/clip/'
    if basedir is 'dt':
        basedir = 'kepdat/DT/'

    basedir = os.path.join(kepdir,basedir)
    g = glob.glob(basedir+'*%09i*.fits' % KIC)
    return g

def mqload(files):
    """
    Load up a table set given a list of fits files.
    """
    tset = atpy.TableSet()
    
    for f in files:
        hdu = pyfits.open(f)
        t = atpy.Table(f,type='fits')
        Q = hdu[0].header['QUARTER']
        t.table_name = 'Q%i' % Q
        t.add_keyword('PATH',f)
        t.add_keyword('QUARTER',Q)
        tset.append(t)

    return tset 

def nQ(tset0):
    """

    Normalize the quarters, this is just for easy viewing.

    """
    tset = copy.deepcopy(tset0)
    offset = ma.masked_invalid(tset[0].SAP_FLUX).mean()
    for t in tset:
        t.add_column('f',t.SAP_FLUX - ma.masked_invalid(t.SAP_FLUX).mean() + 
                     offset)

    return tset    

def sQ(tset0):
    """
    Stitch quarters together.
    """

    tset = copy.deepcopy(tset0)

    f     = [tab.f for tab in tset]
    t     = [tab.TIME for tab in tset]
    cad   = [tab.CADENCENO for tab in tset]

    cad  = detrend.larr(cad)
    f    = detrend.larr(f)
    t    = detrend.larr(t)

    # Figure out which cadences are missing and fill them in.
    cad,iFill = cadFill(cad)
    nFill = cad.size

    fNew = np.empty(nFill)
    tNew = np.empty(nFill)

    tNew[::] = np.nan
    fNew[::] = np.nan

    tNew[iFill] = t
    fNew[iFill] = f

    t = tNew
    f = fNew

    t = ma.masked_invalid(t)
    cad = ma.masked_array(cad)
    cad.mask = t.mask
    sp = UnivariateSpline(cad.compressed(),t.compressed(),s=0,k=1)
    cad = cad.data
    t = sp(cad)

    tLC = atpy.Table()
    tLC.table_name = "LC" 
    tLC.keywords = tset[0].keywords
    tLC.add_column('f',f)
    tLC.add_column('TIME',t)
    tLC.add_column('cad',cad)

    return tLC

def cut(tLC0):
    """
    Cut out the bad regions.
    """

    tLC = copy.deepcopy(tLC0)
    rec = mlab.csv2rec('ranges/cut_time.txt')

    tm = ma.masked_array(tLC.TIME,copy=True)
    for r in rec:
        tm = ma.masked_inside(tm,r['start'],r['stop'])

    f = ma.masked_array(tLC.f,mask=tm.mask,fill_value=np.nan,copy=True)
    tLC.f = f.filled()
        
    return tLC

def outReg(tLC0):
    """
    Outlier rejection.
    """

    tLC = copy.deepcopy(tLC0)
    medf = nd.median_filter(tLC.f,size=4)
    resf = tLC.f - medf
    resf = ma.masked_invalid(resf)
    resfcomp = resf.compressed()
    lo,up = np.percentile(resfcomp,0.1),np.percentile(resfcomp,99.9)
    out = ma.masked_outside(resf,lo,up,copy=True)
    tLC.f[np.where(out.mask)] = np.nan

    tLC.cad,tLC.f = detrend.nanIntrp(tLC.cad,tLC.f,nContig=25)
    return tLC

def prepLC(tset):
    """
    Prepare Lightcurve

    1.  Fill in missing cadences (time, cad, flux nan)
    2.  Cut out bad regions.
    3.  Interpolate over the short gaps in the timeseries.

    """

    tset = nQ(tset)
    tset = cut(tset)
    tLC = sQ(tset)
    tLC = outReg(tLC)

    # Normalize lightcurve.
    tLC.f /= np.median(tLC.f)
    tLC.f -= 1

    # Set time = 0
    tLC.add_column('t',tLC.TIME)
    tLC.t -= np.nanmin(tLC.t)

    tLC.f = detrend.mqclip(tLC.t,tLC.f)
    tLC.cad,tLC.f = detrend.nanIntrp(tLC.cad,tLC.f,nContig=25)

    return tLC




    
def cadFill(cad0):
    """
    Cadence Fill

    We want the elements of the arrays to be evenly sampled so that
    phase folding is equivalent to array reshaping.

    Parameters
    ----------
    cad : Array of cadence identifiers.
    
    Returns
    -------
    cad   : New array of cadences (without gaps).
    iFill : Indecies that were not missing.

    """
    
    bins = np.arange(cad0[0],cad0[-1]+2)
    count,cad = np.histogram(cad0,bins=bins)
    iFill = np.where(count == 1)[0]
    
    return cad,iFill


def iscadFill(t,f):
    """
    Is the time series evenly spaced.

    The vectorized implementation of LDMF depends on the data being
    evenly sampled.  This function checks the time between cadances.
    If this is more than a small fraction of the cadence length,
    fail!
    """

    tol = keptoy.lc/100. 
    return ( (t[1:] - t[:-1]).ptp() < tol ) & (t.size == f.size)

    
