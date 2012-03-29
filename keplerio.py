"""
Functions for facilitating the reading and writing of Kepler files.

Load up a lightcurve
--------------------

>>> files = keplerio.KICPath(8144222,'orig')
>>> tLCset = map(keplerio.qload,files)
>>> tLCset = map(keplerio.nQ,tLCset)
>>> tLCset = atpy.TableSet(tLCset)
>>> tLC = keplerio.prepLC(tLCset)

"""
import numpy as np
from numpy import ma
from scipy.interpolate import UnivariateSpline
from scipy import ndimage as nd
import copy
import atpy
import os
import sys
import tarfile
import glob
import pyfits

from matplotlib.mlab import csv2rec
import keptoy
import detrend
import tfind

kepdir = os.environ['KEPDIR']
kepdat = os.environ['KEPDAT']
cbvdir = os.path.join(kepdir,'CBV/')
kepfiles = os.path.join(os.environ['KEPBASE'],'files')
qsfx = csv2rec(os.path.join(kepfiles,'qsuffix.txt'),delimiter=' ')

def KICPath(KIC, QL=range(1,9) ):
    pathL = []
    for Q in QL:
        tQ = qsfx[ np.where(qsfx['q'] == Q) ]
        path = 'Q%i/kplr%09d-%s_llc.fits' % ( Q,KIC,tQ.suffix[0] ) 
        path = 'archive/data3/privkep/EX/' + path
        pathL.append(path)
    return pathL

def tarXL(filesL):
    """
    Extract List of files from tar archive.
    """
    qfunc = lambda s : int(s.split('Q')[1].split('/kplr')[0])
    QL = map(qfunc,filesL)
    t = atpy.Table()
    t.add_column('file',filesL)
    t.add_column('Q',QL)
    for q in np.unique(QL):
        tQ = t.where(t.Q == q)
        tar  = os.path.join(kepdat,'EX_Q%i.tar' % q)
        nload = 0
        nfail = 0

        try:
            tar  = tarfile.open(tar)
            for file in tQ.data['file']:
                try:
                    tar.extract(file)
                    print "Extracted %s" % file
                    nload +=1 
                except KeyError:
                    print sys.exc_info()[1]
                    nfail +=1
            tar.close()
            print "Loaded %i, Failed %i" %(nload,nfail)
        except IOError:
            print sys.exc_info()[1]

def qload(file):
    """
    Quarter Load

    Takes the fits file from the Kepler team and outputs an atpy table.

    Parameters
    ----------
    file : Path to the .fits file.

    Returns
    -------
    t    : atpy table

    """
    hdu = pyfits.open(file)
    t = atpy.Table(file,type='fits')  

    fm = ma.masked_invalid(t.SAP_FLUX)
    update_column(t,'fmask',fm.mask)

    kw = ['NQ','CUT','OUTREG']
    hkw = ['QUARTER','MODULE','CHANNEL','OUTPUT']

    remcol = []

    t.add_keyword('PATH',file)

    for k in kw:
        t.keywords[k] = False

    for k in hkw:
        t.keywords[k] = hdu[0].header[k]


    t.keywords.pop('TIERABSO') # Problems with conversion

    t.table_name = 'Q%i' % t.keywords['QUARTER']
    return t

def bvload(quarter,module,output):
    """
    Load basis vector.

    """    
    bvfile = os.path.join( cbvdir,'kplr*-q%02d-*.fits' % quarter)
    bvfile = glob.glob(bvfile)[0]
    bvhdu  = pyfits.open(bvfile)
    bvkw   = bvhdu[0].header
    bvcolname = 'MODOUT_%i_%i' % (module,output)
    tBV    = atpy.Table(bvfile,hdu=bvcolname,type='fits')
    return tBV

    
def nQ(t0):
    """
    Normalize lightcurve.

    Parameters
    ----------
    t0 : input table.

    Returns
    -------
    t  : Table with new, normalized columns.
    
    """
    t = copy.deepcopy(t0)

    col   = ['SAP_FLUX','PDCSAP_FLUX']
    ecol  = ['SAP_FLUX_ERR','PDCSAP_FLUX_ERR']
    col2  = ['f','fpdc']   # Names for the modified columns.
    ecol2 = ['ef','efpdc']

    for c,ec,c2,ec2 in zip(col,ecol,col2,ecol2):
        update_column(t,c2, copy.deepcopy(t[c]) )
        update_column(t, ec2, copy.deepcopy(t[ec]) )
        medf = np.median(t[c])
        t.data[c2]  =  t.data[c2]/medf - 1
        t.data[ec2] =  t.data[ec2]/medf

    t.keywords['NQ'] = True

    return t

def isBadReg(t):
    """
    Cut out the bad regions.

    Paramters
    ---------
    t : time

    Returns
    -------
    mask : mask indicating bad values. True is bad.
    """
    cutpath = os.path.join(os.environ['KEPDIR'],'ranges/cut_time.txt')
    rec = atpy.Table(cutpath,type='ascii').data
    tm = ma.masked_array(t,copy=True)
    for r in rec:
        tm = ma.masked_inside(tm,r['start'],r['stop'])
    mask = tm.mask
    return mask

def sQ(tLCset0):
    """
    Stitch Quarters together.

    Fills in missing times and cadences with their proper values.  It
    assigns placeholder values for other columns.
    - floats --> nan
    - bools  --> True

    Parameters
    ----------
    tLCset0 : List of tables to stitch together.
    
    Returns
    -------
    tLC : Lightcurve that has been stitched together.    

    """

    tLCset = copy.deepcopy(tLCset0)
    tLC = atpy.Table()
    tLC.table_name = "LC" 
    tLC.keywords = tLCset[0].keywords

    # Figure out which cadences are missing and fill them in.
    cad       = [tab.CADENCENO for tab in tLCset]
    cad       = np.hstack(cad) 
    cad,iFill = cadFill(cad)
    nFill     = cad.size
    update_column(tLC,'cad',cad)

    for t in tLCset:
        update_column(t,'q',np.zeros(t.data.size) + 
                      t.keywords['QUARTER'] )


    # Add all the columns from the FITS file.
    fitsname = tLCset[0].data.dtype.fields.keys()
    
    for fn in fitsname:
        col = [tab[fn] for tab in tLCset] # Column in list form
        col =  np.hstack(col)       # Convert the list to an array 

        # Fill Value
        if col.dtype is np.dtype('bool'):
            fill_value = True
        else:
            fill_value = np.nan

        ctemp = np.empty(nFill,dtype=col.dtype) # Temporary column
        ctemp[::] = fill_value
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


def isStep(f):
    """
    isStep

    Identify steps in the LC
    """
    stepThrsh = 1e-3 # Threshold steps must be larger than.  
    wd = 4 # The number of cadences we use to determine the change in level.
    medf = nd.median_filter(f,size=wd)

    kern = np.zeros(wd)
    kern[0] = 1
    kern[-1] = -1

    diff   = nd.convolve(medf,kern)
    return np.abs(diff) > stepThrsh

def isOutlier(f):
    """
    Is Outlier

    Identifies single outliers based on a median & percentile filter.

    Parameters
    ----------
    f : Column to perform outlier rejection.

    Returns
    -------
    mask : Boolean array. True is outlier.
    """

    medf = nd.median_filter(f,size=4)
    resf = f - medf
    resf = ma.masked_invalid(resf)
    resfcomp = resf.compressed()
    lo,up = np.percentile(resfcomp,0.1),np.percentile(resfcomp,99.9)
    resf = ma.masked_outside(resf,lo,up,copy=True)
    mask = resf.mask
    return mask

def tcbvdt(tQLC0,fcol,efcol,cadmask=None,dt=False,ver=True):
    """
    Table CBV Detrending

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
    tQLC = copy.deepcopy(tQLC0)
    cbv = [1,2,3,4,5,6] # Which CBVs to use.
    ncbv = len(cbv)

    kw = tQLC.keywords
    assert kw['NQ'],'Assumes lightcurve has been normalized.' 

    cad   = tQLC['CADENCENO' ]
    t     = tQLC['TIME'      ]
    f     = tQLC[fcol        ]
    ferr  = tQLC[efcol       ]

    tBV = bvload(kw['QUARTER'],kw['MODULE'],kw['OUTPUT'])
    bv = np.vstack( [tBV['VECTOR_%i' % i] for i in cbv] )

    # Remove the bad values of f by setting them to nan.
    mask = tQLC.fmask


    fm   = ma.masked_array(f,mask=mask,copy=True)
    fdt  = ma.masked_array(f,mask=mask,copy=True)
    fcbv = ma.masked_array(f,mask=mask,copy=True)
    tm   = ma.masked_array(t,mask=mask,copy=True)
    sL   = detrend.cbvseg(tm)

    for s in sL:
        idnm = np.where(~fm[s].mask)
        a1,a2= detrend.segfitm(t[s],fm[s],bv[:,s])
        fdt[s][idnm]  = a1.astype('>f4') 
        fcbv[s][idnm] = a2.astype('>f4')
    update_column(tQLC,'fdt',fdt.data)
    update_column(tQLC,'fcbv',fcbv.data)

    return tQLC

def ppQ(t0,ver=True):
    """
    Preprocess Quarter

    Apply the following functions to every quarter.
    """
    t = copy.deepcopy(t0)

    update_column(t,'isBadReg',  isBadReg(t.TIME) )
    update_column(t,'isOutlier', isOutlier(t.f) )
    update_column(t,'isStep',    isStep(t.f) )
    t.fmask = t.fmask | t.isBadReg | t.isOutlier | t.isStep
    t.data = tcbvdt(t,'f','ef').data

    return t

def prepLC(tLCset,ver=True):
    """
    Prepare Lightcurve

    1.  Fill in missing cadences (time, cad, flux nan)
    2.  Cut out bad regions.
    3.  Interpolate over the short gaps in the timeseries.

    """
    kw = tLCset.keywords
    
    ppQlam = lambda t0 : ppQ(t0,ver=ver)
    tLCset = map(ppQlam,tLCset)
    tLC    = sQ(tLCset)

    update_column(tLC,'t',tLC.TIME)

    for k in kw.keys():
        tLC.keywords[k] = kw[k]

    f = ma.masked_invalid(tLC.fdt-tLC.fcbv)
    f.fill_value=0

    dM6 = tfind.mtd(tLC.t,f.filled(),tLC.isStep,tLC.fmask,12)
    isNoisey = noiseyReg(tLC.t,dM6)
    update_column(tLC,'isNoisey',isNoisey)

    dM6.mask = dM6.mask | isNoisey
    update_column(tLC,'dM6',dM6)
    update_column(tLC,'dM6mask',dM6.mask)
    
    tLC.fmask = tLC.fmask | tLC.isNoisey

    return tLC

def noiseyReg(t,dM,thresh=2):
    """
    Noisey Region
    
    If certain regions are much noisier than others, we should remove
    them.  A typical value of the noise is computed using a median
    absolute deviation (MAD) on individual regions.  If certain regions are
    noisuer by thresh, we mask them out.

    Parameters
    ----------

    dM     : Single event statistic (masked array)
    thresh : If region has MAD > thresh * typical MAD throw it out.
    """

    tm   = ma.masked_array(t,mask=dM.mask)    
    sL   = detrend.cbvseg(tm)
    
    madseg  = np.array([ma.median(ma.abs( dM[s] ) ) for s in sL])
    madLC   = np.median(madseg)  # Typical MAD of the light curve.

    isNoisey = np.zeros( tm.size ).astype(bool)
    sNL = [] # List to keep track of the noisey segments
    for s,mads in zip(sL,madseg):
        if mads > thresh * madLC:
            isNoisey[s] = True
            sNL.append(s)

    if len(sNL) != 0:
        print "Removed following time ranges because data is noisey"
        print "----------------------------------------------------"

    for s in sNL:
        print "%.2f %.2f " % (t[s.start],t[s.stop-1])

    return isNoisey

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

def update_column(t,name,value):
    try:
        t.add_column(name,value)
    except ValueError:
        t.remove_columns([name])
        t.add_column(name,value)
