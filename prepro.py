"""
Pre-Processing

Process photometry before running the grid search.

>>> Computational Module; No Plotting <<<
"""
import os

import numpy as np
from numpy import ma
from scipy import ndimage as nd
from matplotlib import mlab
from matplotlib.mlab import csv2rec,rec_append_fields
import atpy  # Could get rid of this, just need to change cutpath
import pyfits
import h5py

import cotrend
import config
import detrend
import h5plus
import keplerio
import tfind

kepdir     = os.environ['KEPDIR']
cbvdir     = os.path.join(kepdir,'CBV/')
kepfiles   = os.path.join(os.environ['KEPBASE'],'files')
qsfx       = csv2rec(os.path.join(kepfiles,'qsuffix.txt'),delimiter=' ')
sapkeypath = os.path.join(kepfiles,'sap_quality_bits.txt')
sapkey     = csv2rec(sapkeypath,delimiter=' ')
sapdtype   = zip( sapkey['key'],[bool]*sapkey['key'].size )
cutpath    = os.path.join(kepfiles,'ranges/cut_time.txt')
cutList    = atpy.Table(cutpath,type='ascii').data

def rec_zip(rL):
    """
    """
    ro =rL[0]
    for i in range(1,len(rL)):
        fields = list(rL[i].dtype.names)
        vals   = [rL[i][f] for f in fields ]
        ro = mlab.rec_append_fields(ro,fields,vals)
    return ro

class Lightcurve(h5plus.File):
    def raw(self,files):
        """
        Take list of .fits files and store them in the raw group
        """
        raw  = self.create_group('/raw')
        hduL = []
        kicL = []
        qL   = []
        for f in files:
            h = pyfits.open(f)
            hduL += [h]
            kicL += [h[0].header['KEPLERID'] ]
            qL   += [h[0].header['QUARTER'] ]

        assert np.unique(kicL).size == 1,'KEPLERID not the same'
        assert np.unique(qL).size == len(qL),'duplicate quarters'

        self.attrs['KEPLERID'] = kicL[0] 
        for h,q in zip(hduL,qL):
            r = np.array(h[1].data)
            r = modcols(r)
            raw['Q%i'  % q] = np.array(r)
            
    def dt(self):
        """
        Iterates over the quarters stored in raw
        
        Applies the detrending
        """
        raw = self['/raw']
        dt  = self.create_group('/dt')
        for item in raw.items():
            quarter = item[0]
            ds      = item[1]
            r = ds[:]
            rawFields = list(r.dtype.names)            

            r = rqmask(r)
            r = rdt(r)               
            dtFields = list(r.dtype.names)
            dtFields = [f for f in dtFields if rawFields.count(f) ==0]
            dt[quarter] = mlab.rec_keep_fields(r,dtFields)

    def cal(self,svd_folder):
        """
        Wrap over the calibration step

        Parameters
        ----------

        svd_folder : Where to find the SVD matricies.  They must be
                     named in the following fashion Q1.svd.h5,
                     Q2.svd.h5 ...
        """
        dt = self['/dt']
        cal = self.create_group('/cal')
        for item in dt.items():
            quarter = item[0]
            ds      = item[1]
            rdt     = ds[:]
            dtFields = list(rdt.dtype.names)
            
            
            fdt = ma.masked_array(rdt['fdt'],rdt['fmask'])
            hsvd = h5py.File(os.path.join(svd_folder,'%s.svd.h5' % quarter))

            bv   = hsvd['V'][:config.nMode]
            fit    = ma.zeros(fdt.shape)
            p1,fit = cotrend.bvfitm(fdt.astype(float),bv)
            fcal  = fdt - fit
            rcal = mlab.rec_append_fields(ds[:],['fit','fcal'],[fit,fcal])

            calFields = list(rcal.dtype.names)
            calFields = [f for f in calFields if dtFields.count(f) ==0]

            cal[quarter] = mlab.rec_keep_fields(rcal,calFields)

    def mqcal(self):
        """
        Stitch Quarters
        """
        raw   = self['raw']
        dt    = self['dt']
        cal   = self['cal']
        
        rL = []
        for item in cal.items():
            quarter = item[0]
            ds      = item[1]
            r = rec_zip([ raw[quarter],dt[quarter],cal[quarter] ] )
            rL.append(r)

        rLC = keplerio.rsQ(rL)
        binlen = [3,6,12]
        for b in binlen:
            bcad = 2*b
            fcal = ma.masked_array(rLC['fcal'],rLC['fmask'])
            dM = tfind.mtd(rLC['t'],fcal,bcad)
            rLC = mlab.rec_append_fields(rLC,'dM%i' % b,dM.filled() )
        self['mqcal'] = rLC

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
    for k in config.cadGrow.keys():
        nGrow = config.cadGrow[k] * 2 # Expand in both directions
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

    medf = nd.median_filter(f,size=config.wd)

    kern = np.zeros(config.wd)
    kern[0] = 1
    kern[-1] = -1

    diff   = nd.convolve(medf,kern)
    return np.abs(diff) > config.stepThrsh

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
