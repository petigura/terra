"""
Functions for facilitating the reading and writing of Kepler files.
"""
import numpy as np
from numpy import ma
from scipy.interpolate import UnivariateSpline
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

def update_column(t,name,value):
    try:
        t.add_column(name,value)
    except ValueError:
        t.remove_columns([name])
        t.add_column(name,value)

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
    kw = ['nQ','cut','outreg']
    
    for f in files:
        hdu = pyfits.open(f)
        t = atpy.Table(f,type='fits')

        t.keywords = dict(hdu[0].header)
        t.add_keyword('PATH',f)
        for k in kw:
            t.keywords[k] = False

        t.table_name = 'Q%i' % t.keywords['QUARTER']        
        tset.append(t)

    return tset 

def nQ(tset0):
    """
    Normalize the quarters.  

    """
    tset = copy.deepcopy(tset0)

    col = ['SAP_FLUX','PDCSAP_FLUX']
    ecol = ['SAP_FLUX_ERR','PDCSAP_FLUX_ERR']
    col2 = ['f','fpdc']    # Names for the modified columns.
    ecol2 = ['ef','efpdc']

    for c,ec,c2,ec2 in zip(col,ecol,col2,ecol2):
        for t in tset:
            update_column(t,c2, copy.deepcopy(t[c]) )
            update_column(t, ec2, copy.deepcopy(t[ec]) )

            medf = np.median(t[c])
            t.data[c2]  =  t.data[c2]/medf - 1
            t.data[ec2] =  t.data[ec2]/medf
            t.keywords['nQ'] = True


    return tset  

def cut(tLC0,cutk=['f','ef','fpdc','efpdc'] ):
    """
    Cut out the bad regions.
    """
    tLC = copy.deepcopy(tLC0)
    cutpath = os.path.join(os.environ['KEPDIR'],'ranges/cut_time.txt')
    rec = atpy.Table(cutpath,type='ascii').data
    tm = ma.masked_array(tLC.TIME,copy=True)
    for r in rec:
        tm = ma.masked_inside(tm,r['start'],r['stop'])

    for k in cutk:
        tLC.data[k][np.where(tm.mask)] = np.nan

    tLC.keywords['cut'] = True
    return tLC

def sQ(tLCset0):
    """
    Stitch quarters together.

    Fills in missing times and cadences with their proper values
    """

    tLCset = copy.deepcopy(tLCset0)
    tLC = atpy.Table()
    tLC.table_name = "LC" 
    tLC.keywords = tLCset[0].keywords

    # Figure out which cadences are missing and fill them in.
    cad       = [tab.CADENCENO for tab in tLCset]
    cad       = detrend.larr(cad)       # Convert the list to an array 
    cad,iFill = cadFill(cad)
    nFill     = cad.size
    update_column(tLC,'cad',cad)

    for t in tLCset:
        update_column(t,'q',np.zeros(t.data.size) + 
                      t.keywords['QUARTER'] )


    # Add all the columns from the FITS file.
    fitsname = tLCset[0].data.dtype.fields.keys()
    for fn in fitsname:
        ctemp = np.empty(nFill) # Temporary column
        ctemp[::] = np.nan      # default value is nan

        col = [tab[fn] for tab in tLCset] # Column in list form
        col =  detrend.larr(col)       # Convert the list to an array 

        ctemp[iFill] = col
        update_column(tLC,fn,ctemp)


    # Fill in the missing times.
    t        = ma.masked_invalid(tLC['TIME'])
    cad      = ma.masked_array(cad)
    cad.mask = t.mask
    sp       = UnivariateSpline(cad.compressed(),t.compressed(),s=0,k=1)
    cad      = cad.data
    tLC.data['TIME'] = sp(cad)

    return tLC

def outReg(f):
    """
    Outlier rejection.

    Reject single outliers based on a median & percentile filter.  sQ
    need to be run.

    Parameters
    ----------
    f : Column to perform outlier rejection.
    """

    medf = nd.median_filter(f,size=4)
    resf = f - medf
    resf = ma.masked_invalid(resf)
    resfcomp = resf.compressed()
    lo,up = np.percentile(resfcomp,0.1),np.percentile(resfcomp,99.9)
    out = ma.masked_outside(resf,lo,up,copy=True)
    f[np.where(out.mask)] = np.nan

    return f

def toutReg(tLC0,outregcol=['f','fpdc']):
    tLC = copy.deepcopy(tLC0)
    for orc in outregcol:
        x = outReg( tLC.data[orc] )

        eorc = 'e'+orc
        ex = outReg( tLC.data[orc] )

        cad,x = detrend.nanIntrp(tLC.data['CADENCENO'],x,nContig=25)
        cad,ex = detrend.nanIntrp(tLC.data['CADENCENO'],ex,nContig=25)

        tLC.data[orc] = x
        tLC.data[eorc] = ex


    return tLC
 
def prepLC(tLCset):
    """
    Prepare Lightcurve

    1.  Fill in missing cadences (time, cad, flux nan)
    2.  Cut out bad regions.
    3.  Interpolate over the short gaps in the timeseries.

    """

    tLCset = nQ(tLCset)
    for tLC in tLCset:
        tLC.data = cut(tLC).data
        tLC.data = toutReg(tLC).data
        data,ffit,bvectors,p1 = detrend.cbv(tLC,'f','ef')
        update_column(tLC,'fcbv',ffit)

    tLC = sQ(tLCset)

    # Set time = 0
    update_column(tLC,'t',tLC.TIME)
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

    
