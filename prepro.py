"""
Pre-Processing

Process photometry before running the grid search.
"""
import numpy as np
from numpy import ma
from scipy import ndimage as nd

import atpy
import keplerio
import copy
import os
from matplotlib.mlab import csv2rec
import glob
import pyfits


import detrend
import tfind
from keplerio import update_column

kepdir = os.environ['KEPDIR']
kepdat = os.environ['KEPDAT']
cbvdir = os.path.join(kepdir,'CBV/')
kepfiles = os.path.join(os.environ['KEPBASE'],'files')
qsfx = csv2rec(os.path.join(kepfiles,'qsuffix.txt'),delimiter=' ')

def prepLC(tLC):
    """
    Take raw photometry and prepare it for the grid search.

    This function is called by the prepro script.

    2.  Cut out bad regions.
    3.  Interpolate over the short gaps in the timeseries.

    """
    ##################
    # Build the mask #
    ##################

    tLC = mapq(qmask,tLC)

    ##############################################
    # Detrend the light curve quarter by quarter #
    ##############################################

    def cbvfunc(t):
        # Figure out what quarter we're dealing with
        q = np.unique(t.q)[0]
        return tcbvdt(t,'f','ef',q)
        
    tLC = mapq(cbvfunc,tLC)

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

def mapq(func,tLC):
    """
    Map to Quarter

    Breaks apart the tLC on the q column and applys a function to each
    quarter individually.  Then it repackages it up with sQ.    
    """

    qL = np.sort(np.unique(tLC.q))
    qL = qL[~np.isnan(qL)] # remove the nans
    tLCtemp = []

    for q in qL:
        tQ = copy.deepcopy( tLC.where( tLC.q == q) )
        tQ = func(tQ)
        tLCtemp.append(tQ)

    tLC = keplerio.sQ(tLCtemp)
    return tLC

def tcbvdt(tQLC0,fcol,efcol,q,cadmask=None,dt=False,ver=True):
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

    t     = tQLC['t'  ]
    f     = tQLC[fcol ]
    ferr  = tQLC[efcol]

    mod,out = keplerio.idQ2mo(kw['KEPLERID'],q)
    tBV = bvload(q,mod,out)
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

def qmask(t0):
    """
    Quarter mask

    Determine the masked regions for the quarter.
    """
    t = copy.deepcopy(t0)

    update_column(t,'isBadReg',  isBadReg(t.t) )
    update_column(t,'isOutlier', isOutlier(t.f) )
    update_column(t,'isStep',    isStep(t.f) )
    update_column(t,'isDis',     isDis(t.SAP_QUALITY) )

    fm = ma.masked_invalid(t.SAP_FLUX)
    fm.mask = fm.mask | t.isBadReg | t.isOutlier | t.isStep | t.isDis
    update_column(t,'fmask',fm.mask)

    return t

def isDis(qual):
    """
    Cut out discontinuities

    Parameters
    ----------
    qual - SAP_QUALITY
    """

    preCut  = 4 # Number of cadences to cut out before discontinuity
    postCut = 50 # Number of cadences to cut after discont

    cad = np.arange(qual.size)
    cad = ma.masked_array(cad)
    disId = cad[qual==1024]
    for id in disId:
        cad = ma.masked_inside(cad,id-preCut,id+postCut)

    return cad.mask

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

def noiseyReg(t,dM,thresh=2):
    """
    Noisey Region
    
    If certain regions are much noisier than others, we should remove
    them.  A typical value of the noise is computed using a median
    absolute deviation (MAD) on individual regions.  If certain regions are
    noiser by thresh, we mask them out.

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
