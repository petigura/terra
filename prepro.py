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

from matplotlib import mlab
from matplotlib.mlab import csv2rec,rec_append_fields
import glob
import pyfits

import cotrend
import detrend
import tfind
from keplerio import update_column
import qalg
from config import stepThrsh,wd,cadGrow

kepdir = os.environ['KEPDIR']
cbvdir = os.path.join(kepdir,'CBV/')
kepfiles = os.path.join(os.environ['KEPBASE'],'files')
qsfx = csv2rec(os.path.join(kepfiles,'qsuffix.txt'),delimiter=' ')
sapkeypath = os.path.join(kepfiles,'sap_quality_bits.txt')
sapkey = csv2rec(sapkeypath,delimiter=' ')
sapdtype = zip( sapkey['key'],[bool]*sapkey['key'].size )

cutpath = os.path.join(kepfiles,'ranges/cut_time.txt')
cutList = atpy.Table(cutpath,type='ascii').data

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
    minseg = 100
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

    mask = tQLC.fmask

    fm   = ma.masked_array(f,mask=mask,copy=True)
    fdt  = ma.masked_array(f,mask=mask,copy=True)
    fcbv = ma.masked_array(f,mask=mask,copy=True)
    tm   = ma.masked_array(t,mask=mask,copy=True)

    # Detrend each each CBV segment individually
    label   = detrend.sepseg(tm)
    sL   = ma.notmasked_contiguous(label)
    sL   = [s for s in sL if s.stop-s.start > minseg]
    for s in sL:
        a1,a2= cotrend.dtbvfitm(t[s],fm[s],bv[:,s])
        fdt[s]  = a1.astype('>f4') 
        fcbv[s] = a2.astype('>f4')

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


def rdt(r0):
    """
    Detrend light curve

    Parameters
    ----------
    r0 : with `f`, `fmask`, `t`, `segEnd` fields

    Returns
    -------
    r  : same record array with the following fields:
         label - 0,1,2 identifies groups for spline detrending.
         ftnd  - the best fit trend
         fdt   - f - ftnd.
    """

    r = r0.copy()
    # Detrend the flux
    fm = ma.masked_array(r['f'],r['fmask'])
    tm = ma.masked_array(r['t'],r['fmask'])

    # Assign a label to the segEnd segment
    label = ma.masked_array( np.zeros(r.size)-1, r['segEnd'] )

    ftnd = fm.copy()

    sL = ma.notmasked_contiguous(label)
    nseg = len(sL)
    for i in range(nseg):
        s = sL[i]
        ftnd[s]  = detrend.spldtm(tm[s],fm[s])
        label[s] = i

    r   = mlab.rec_append_fields(r,'label',label.data)
    fdt = fm-ftnd

    r = mlab.rec_append_fields(r,'ftnd',ftnd.data)
    r = mlab.rec_append_fields(r,'fdt',fdt.data)
    return r

def modcols(r0):
    """
    Modify Columns

    1. Changes TIME, CADENCENO to t, cad
    2. rnQ      - normalize quarter
    3. rnanTime - remove nans from time series
    """

    r = r0.copy()
    oldName = ['TIME','CADENCENO']
    newName = ['t','cad']
    for o,n in zip(oldName,newName):
        r = mlab.rec_append_fields(r,n,r[o])
        r = mlab.rec_drop_fields(r,o)

    r = keplerio.rnQ(r)
    r = keplerio.rnanTime(r)
    return r

def qmask(t0):
    """
    Quarter mask

    Determine the masked regions for the quarter.
    """
    t = copy.deepcopy(t0)
    r = rqmask(t.data)
    t = qalg.rec2tab(r)
    t.keywords = t0.keywords
    return t


def rqmask(r0):
    """
    Record Array Quarter Mask

    Computes two masks:
    - fmask  : True if a certain cadence is to be excluded from the
               lightcurve
    - segEnd : True if this is a region between two detrending
               segments.  These occur at discontinuities.  The
               detrender should not try to run over a jump in the
               photomtry.
    
    Parameters
    ----------
    r0 : lightcurve record array with the following fields:
         - `t`
         - `f`      
         - `SAP_QUALITY`

    Returns
    -------
    r  : record array with the following fields added:
         - fmask
         - segEnd
         - isStep
         - isOutlier
         - isBadReg
    """
    r = r0.copy()

    rqual = parseQual(r['SAP_QUALITY'])

    r = rec_append_fields(r,'isBadReg' ,isBadReg(r['t'])  )
    r = rec_append_fields(r,'isOutlier',isOutlier(r['f']) )
    r = rec_append_fields(r,'isStep'   ,isStep(r['f'])    )
    r = rec_append_fields(r,'isDis'    ,isStep(rqual['dis'])  )
    r = rec_append_fields(r,'desat'    ,rqual['desat']  )
    r = rec_append_fields(r,'atTwk'    ,rqual['atTwk']  )


    # Grow masks for bad regions
    for k in cadGrow.keys():
        nGrow = cadGrow[k] * 2 # Expand in both directions
        b = r[k]
        b = nd.convolve( b.astype(float) , np.ones(nGrow,float) )
        r[k] = b > 0

    # fmask is the union of all the individually masked elements    
    fm = ma.masked_invalid(r.SAP_FLUX)
    fm.mask = \
        fm.mask | r['isBadReg'] | r['isOutlier'] | r['isStep'] | \
        r['isDis'] | r['desat'] | r['atTwk']
    
    r = rec_append_fields(r,'fmask'    ,fm.mask)
    
    # segEnd is the union of the manually excluded regions and the
    # attitude tweaks.
    segEnd = r['isBadReg'] | r['atTwk'] | r['isStep']
    r = rec_append_fields(r,'segEnd',segEnd)

    return r


def parseQual(SAP_QUALITY):
    """
    Parse the SAP_QUALITY field

    Parameters
    ----------
    SAP_QUALITY - Quality flag set from fits file.

    Returns
    -------
    Record array where each of the fields are booleans corresponding
    to wheter that particular bin was set.

    rqual['Dis'] - True if the Kepler team identified a discontinuity.

    """

    rqual = np.zeros(SAP_QUALITY.size,dtype=sapdtype)
    for b,k in zip(sapkey['bit'],sapkey['key']):
        val = 2**(b-1)
        rqual[k] = (SAP_QUALITY & val) != 0

    return rqual

def isDis(bDis):
    """
    Is discontinuity?

    If the kepler team identifies a discontinuity.  We mask out
    several cadences before, and many cadences afterward.

    Parameters
    ----------
    bDis - Boolean from Kepler SAP_QAULITY

    Returns
    -------
    Boolean array corresponding to masked out region.

    """
    preCut  = 4 # Number of cadences to cut out before discontinuity
    postCut = 50 # Number of cadences to cut after discont

    cad = np.arange(bDis.size)
    cad = ma.masked_array(cad)
    disId = cad[ bDis ]
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


    tm = ma.masked_array(t,copy=True)
    for r in cutList:
        tm = ma.masked_inside(tm,r['start'],r['stop'])
    mask = tm.mask
    return mask

def isStep(f):
    """
    Is Step

    Identify steps in the LC
    """
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
    good = ~np.isnan(resf)
    resfcomp = resf[good]
    lo,up = np.percentile(resfcomp,0.1),np.percentile(resfcomp,99.9)    
    return good & ( (resf > up) | (resf < lo) )

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

    tm    = ma.masked_array(t,mask=dM.mask)    
    label = detrend.sepseg(tm)
    sL    = ma.notmasked_contiguous(label)

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
