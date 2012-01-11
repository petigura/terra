"""
Functions for facilitating the reading and writing of Kepler files.
"""
import numpy as np
from numpy import ma
from scipy.interpolate import UnivariateSpline
from matplotlib import mlab

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


def prepLC(tLC0):
    """
    Prepare Lightcurve

    1.  Fill in missing cadences (time, cad, flux nan)
    2.  Cut out bad regions.
    3.  Interpolate over the short gaps in the timeseries.

    """

    f     = [tab.SAP_FLUX for tab in tLC0]
    t     = [tab.TIME for tab in tLC0]
    cad   = [tab.CADENCENO for tab in tLC0]


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

    # Cut out the bad regions.
    rec = mlab.csv2rec('ranges/cut_time.txt')

    tm = ma.masked_array(t)
    for r in rec:
        tm = ma.masked_inside(tm,r['start'],r['stop'])
    f = ma.masked_array(f,mask=tm.mask,fill_value=np.nan)
    f = f.filled()

    # Normalize lightcurve.
    f /= np.median(f)
    f -= 1

    # Set time = 0
    t -= np.nanmin(t)

    f = detrend.mqclip(t,f)
    cad,f = detrend.nanIntrp(cad,f,nContig=25)
    
    tLC = atpy.Table()

    tLC.table_name = "LC" 
    tLC.keywords = tLC0[0].keywords
    tLC.add_column('f',f)
    tLC.add_column('t',t)

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

    
