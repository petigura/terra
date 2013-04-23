"""
Pre-Processing

Process photometry before running the grid search.

>>> Computational Module; No Plotting <<<
"""
import os

import numpy as np
np.set_printoptions(precision=2,threshold=20)

from numpy import ma
from scipy import ndimage as nd
from matplotlib import mlab
from matplotlib.mlab import csv2rec,rec_append_fields
import pyfits
import h5py
import pandas 

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
cutList    = pandas.read_csv(cutpath,comment='#').to_records(index=False)

def rec_zip(rL):
    """
    """
    ro =rL[0]
    for i in range(1,len(rL)):
        fields = list(rL[i].dtype.names)
        vals   = [rL[i][f] for f in fields ]
        ro = mlab.rec_append_fields(ro,fields,vals)
    return ro


def raw(h5,files,fields=[]):
    """
    Take list of .fits files and store them in the raw group

    fields - list of fields to keep. Use a subset for smaller file size.
    """
    raw  = h5.create_group('/raw')
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

    h5.attrs['KEPLERID'] = kicL[0] 
    for h,q in zip(hduL,qL):
        r = np.array(h[1].data)
        r = modcols(r)
        raw['Q%i'  % q] = r
        if fields!=[]:
            r = mlab.rec_keep_fields(r,fields)

def mapgroup(h5,f,inpG,outG,**kwd):
    """
    Map function over groups

    Parameters
    ----------
    f   :  function.  must have the following signature
           rout = f(*args)
           where rout is an array and
           args is a list of arrays

    inpG : Input group name (or sequence of names)
    outG : Output group name.
    kwd  : dictionary of keyword dictionaries. Method by which
           we pass non-record array arguments to `f`

    """

    if inpG==outG:
        same=True
    else:
        same=False
    if type(inpG)==str:
        inpG = [inpG]

    qL = [i[0] for i in h5[inpG[0]].items() ]

    for q in qL:
        rinpL = [h5[i][q][:] for i in inpG]

        kwargs={}
        if kwd!={}:
            kwargs  = kwd[q]

        rout    = f(*rinpL,**kwargs)

        if same:
            del h5[outG][q]
        h5[outG][q] = rout


def mask(h5):
    mapgroup(h5,rqmask,'raw','raw')

def dt(h5):
    """
    Iterates over the quarters stored in raw

    Applies the detrending
    """
    raw = h5['/raw']
    h5.create_group('/pp/dt')
    mapgroup(h5,qdt,'raw','/pp/dt')

def cal(h5,svd_folder):
    """
    Wrap over the calibration step

    Parameters
    ----------

    par : dictionary, must contain follo
    svd_folder : Where to find the SVD matricies.  They must be
                 named in the following fashion Q1.svd.h5,
                 Q2.svd.h5 ...
    """
    raw = h5['/raw']
    pp  = h5['/pp']

    dt  = h5['/pp/dt']
    cal = h5.create_group('/pp/cal')

    kwd = {}
    for i in raw.items():
        q = i[0]
        svdpath = os.path.join(svd_folder,'%s.svd.h5' % q)
        kwd[q] = dict(svdpath=svdpath)
    mapgroup(h5,qcal,['raw','/pp/dt'],'/pp/cal',**kwd)


def sQ(h5):
    """
    Stitch Quarters

    Look at all of the groups. Zip all of the column together and
    stich the quarters together.

    Adds the mqcal dataset to the h5 directory.
    """
    
    groups = [ h5[k] for k in '/raw,/pp/dt,/pp/cal'.split(',') ]
    groups = [ g for g in groups if g.name!='/mqcal' ]

    quarters = [i[0] for i in groups[0].items()]

    rL = []
    for q in quarters:
        dsL = rec_zip([ g[q] for g in groups ])
        rL.append(dsL)            

    if len(quarters)==1:
        print "sQ: Only 1 quarter"
        rLC = rL[0]
    else:
        rLC = keplerio.rsQ(rL)

    binlen = [3,6,12]
    try:
        list(rLC.dtype.names).index('fcal')
        for b in binlen:
            bcad = 2*b
            fcal = ma.masked_array(rLC['fcal'],rLC['fmask'])
            dM = tfind.mtd(rLC['t'],fcal,bcad)
            rLC = mlab.rec_append_fields(rLC,'dM%i' % b,dM.filled() )
    except ValueError:
        pass
    h5['/pp/mqcal'] = rLC          

def getseg(lc):
    seglabel = np.zeros(lc.size) - 1
    t = lc['t']
    tm = ma.masked_array(t)

    for i in range( len(cutList)-1 ):
        rng  = cutList[i]
        rng1 = cutList[i+1]

        tm = ma.masked_inside(tm,rng['start'],rng['stop'])

        b = ( tm > rng['stop'] ) & ( tm < rng1['start'] ) 
        seglabel[b] = i
    return seglabel

def rdt(r0):
    """
    Detrend light curve with GP-based detrending.

    Parameters
    ----------
    r0 : with `f`, `fmask`, `t` fields

    Returns
    -------
    r  : same record array with the following fields:
         label - 0,1,2 identifies groups for spline detrending.
         ftnd  - the best fit trend
         fdt   - f - ftnd.
    """
    r = r0.copy()

    seglabel = getseg(r)

    sL = np.unique(seglabel)
    sL = sL[sL >=0]

    fm   = ma.masked_array(r['f'],r['fmask'])
    ftnd = fm.copy()
    
    segstr = ''
    for s in sL:
        b   = seglabel==s
        r2  = r[b]
        t   = r2['t']
        segstr += '%.2f %.2f ' % (t[0],t[1])

        x,y = detrend.bin(r2)           # Compute GP using binned lc (speed)
        yi  = detrend.GPdt(t,x,y) # evaluate at all points
        ftnd[b] = yi 
    print segstr
        
    # Assign a label to the segEnd segment
    label = ma.masked_array( np.zeros(r.size)-1, seglabel )
    fdt   = fm - ftnd

    r = mlab.rec_append_fields(r,'label',label.data)
    r = mlab.rec_append_fields(r,'ftnd',ftnd.data)
    r = mlab.rec_append_fields(r,'fdt',fdt.data)
    return r

def qdt(r):
    """
    Small wrapper around rdt that removes duplicate names
    """
    
    rawFields = list(r.dtype.names)            
    r = rdt(r)               
    dtFields = list(r.dtype.names)
    dtFields = [f for f in dtFields if rawFields.count(f) ==0]
    return mlab.rec_keep_fields(r,dtFields)

def qcal(rraw,rdt,svdpath=None):
    fmask = rraw['fmask']
    fdt = ma.masked_array(rdt['fdt'],fmask)

    hsvd = h5py.File(svdpath,'r')
    bv   = hsvd['V'][:config.nMode]
    hsvd.close()

    fit    = ma.zeros(fdt.shape)
    p1,fit = cotrend.bvfitm(fdt.astype(float),bv)
    fcal  = fdt - fit
    rcal = np.array(zip(fit.data,fcal.data),
                    dtype=[('fit', '<f8'), ('fcal', '<f8')]  )
    return rcal


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
    
    # first mask out the nans
    fmask = ma.masked_invalid(r['f']).mask
    r = rec_append_fields(r,'fmask'    ,fmask)


    r = rec_append_fields(r,'isBadReg' ,isBadReg(r['t'])  )
    r = rec_append_fields(r,'isOutlier',isOutlier(r['f']) )
    fmask = fmask | r['isBadReg'] | r['isOutlier'] 


    r = rec_append_fields(r,'isStep'   ,isStep(r,fmask)[1]    )

    rqual = parseQual(r['SAP_QUALITY'])
    r = rec_append_fields(r,'desat'    ,rqual['desat']  )
    r = rec_append_fields(r,'atTwk'    ,rqual['atTwk']  )

    # Grow masks for bad regions
    for k in config.cadGrow.keys():
        nGrow = config.cadGrow[k] * 2 # Expand in both directions
        b = r[k]
        b = nd.convolve( b.astype(float) , np.ones(nGrow,float) )
        r[k] = b > 0

    # fmask is the union of all the individually masked elements    
    fmask      = fmask | r['isStep'] | r['desat'] | r['atTwk']
    r['fmask'] = fmask
    
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

def isStep(r,fmask):
    """
    Is Step?

    Identify steps in light curve. I compute the running median in
    bins of `size` along the light curve, treating the masked values
    and the nans gracefully. Then I look for derivatives over
    `stepscale`. If the derivative is more than 10 MAD away from the
    typical values, we flag it as a discont.
    """
    size=50
    stepscale = 20

    fm = ma.masked_array(r['f'],fmask)
    fm = ma.masked_invalid(fm)

    # Perform the running median. Yes, this is memory
    # inefficient. However, this is a small portion of runtime.

    fm2D = fm[np.arange(size)[:,np.newaxis] + np.arange(fm.size-size)]
    fmed = ma.median(fm2D,axis=0)
    fmed.mask = fmed.mask | (~fm2D.mask).sum(axis=0) < 20

    # step statistic (positive values are downward steps
    # if step statistic is more than 10 MADs above normal, remove.
    step = fmed[:-stepscale] - fmed[stepscale:]  
    b    = np.abs(step) > 7*ma.median(np.abs(step)) 

    isStep = np.zeros(r['t'].size).astype(bool)
    tmask = r['t'][(size+stepscale)/2:][b.data & ~b.mask]
    if tmask.size > 0:
        print "identified step discont at"
        print tmask
        print "masking out ",tmask.size
        for t in tmask:
            isStep = isStep | ma.masked_inside(r['t'],t-0.5,t+2).mask
            
    return step,isStep

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
