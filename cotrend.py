"""
Erik's cotrending functions.
"""
from scipy import ndimage as nd
from scipy import stats
import glob

import tfind
import detrend
import keplerio
from scipy import optimize
import ebls

from matplotlib import mlab

from numpy import ma,rec
import numpy as np

from config import nMode,sigOut,maxIt

def dtbvfitm(t,fm,bv):
    """
    Detrended Basis Vector Fit
    
    Parameters
    ----------
    t   : time 
    fm  : flux for a particular segment
    bv  : vstack of basis vectors

    Returns
    -------
    fdt  : Detrended Flux
    fcbv : Fit to the detrended flux

    """
    ncbv  = bv.shape[0]
    
    tm = ma.masked_array(t,copy=True,mask=fm.mask)
    mask  = fm.mask 

    bv = ma.masked_array(bv)
    bv.mask = np.tile(mask, (ncbv,1) )

    # Detrend the basis vectors
    bvtnd = [detrend.spldtm(tm,bv[i,:]) for i in range(bv.shape[0]) ] 
    bvtnd = np.vstack(bvtnd)
    bvdt  = bv-bvtnd

    # Detrend the flux
    ftnd  = detrend.spldtm(tm,fm)
    fdt   = fm-ftnd

    p1,fcbv = bvfitm(fdt,bvdt)

    return fdt,fcbv

def bvfitm(fm,bv):
    """
    Basis Vector Fit, masked

    Parameters
    ----------

    fm   : photometry masked.  If there is no, mask we deem every point good.
    bv   : basis vectors, stacked row-wise

    Returns
    -------
    p1   : best fit coeff
    fcbv : fit using a the CBVs

    """
    assert fm.dtype == bv.dtype,\
        'Light curve and basis vectors must be same type' 

    if type(fm) != np.ma.core.MaskedArray:
        fm      = ma.masked_array(fm)
        fm.mask = np.zeros(fm.size).astype(bool)

    assert fm.size == bv.shape[1],"fm and bv must have equal length"
    mask  = fm.mask 

    p1          = np.linalg.lstsq( bv[:,~mask].T , fm[~mask] )[0]
    fcbv        = fm.copy()
    fcbv[~mask] = np.dot( p1 , bv[:,~mask] )

    return p1,fcbv


def robustSVD(D,nMode=nMode,sigOut=sigOut,maxIt=maxIt):
    """
    Robust SVD

    D = U S V.T

    This decomposition is computed using SVD.  SVD is sensitive to
    outliers, so we perform iterative sigma clipping in the `\chi^2`
    sense, but also in the distribution of best fit parameters.

    D_fit[i] = a_1 V_1 + a_2 V_2 + ... a_nMode V_nMode
    
    nMode is the (small) number of principle components we wish to fit
    our data with.

    If any of the a_j, or \chi^2_i = sum((D_fit[i] - D[i])**2)/D.size
    is an outlier, remove that row from D.

    Parameters
    ----------    
    D      : Data matrix.  1-D vectors stacked vertically (row-wise). May
             not contain nans, or masked values.
    nMode  : Number of modes
    sigOut : Clip outliers that are more than sigOut away from the
             median.  Defaults to the config value.
    maxIt  : Maximum number of iterations to perform before exiting.
             Defaults to the config value.


    """
    Dnrow,Dncol = D.shape     
    D    = D.copy()
    gRow = np.ones(Dnrow,dtype=bool) # Good rows (not outliers)

    goodid  = np.arange(D.shape[0])


    # Iterate SVD fits.
    count = 0 
    finished = False
    while finished is False:
        print count
        if count == maxIt:
            finished=True

        D = D[gRow]
        Dnrow,Dncol = D.shape     

        U, s, V = np.linalg.svd(D,full_matrices=False)
        S = np.zeros(V.shape)
        S[:Dnrow,:Dnrow] = np.diag(s)
        
        A    = np.dot(U,S)                  # A is matrix of best fit coeff
        A    = A[:,:nMode]
        Dfit = np.dot(A,V[:nMode])  # Dfit is D represented by a
                                            # trucated series of modes
        
        # Evaluate Chi2
        X2 = np.sum( (Dfit - D)**2,axis=1) / Dncol

        rL = moments(A)
        print "Moments of principle component weight"
        print mlab.rec2txt(rL)        

        # Determine which rows of D are outliers
        dev  = (A - rL['med'])/rL['mad']

        # nMode x Dncol matrix of outlier coeffients
        Aout = abs(dev) > sigOut 

        # Dncol matrix of red-Chi2 outliers
        Xout = (X2 > 3)  | (X2 < 0.5) 
        Xout = Xout.reshape(Xout.size,1)
        
        # Dncol matrix with the number of coeff that failed.
        out    = np.hstack([Xout,Aout])

        # All coefficients must be inliers
        gRow   = out.astype(int).sum(axis=1) == 0 

        # If there are no outliers or we've reached the max number of
        # iterations return the input U,S,V,goodid,X2. If not, clip off
        # the outliers and repeat.
        
        if gRow.all() or (count == maxIt): 
            finished = True
        else:
            names = ['ID'] + ['Chi2'] + ['a%i' % (i+1) for i in range(nMode)]
            dtype = zip(names,[float]*len(names))
            routData = np.hstack([np.vstack(goodid),np.vstack(X2),dev]) 
            routData = [tuple(r) for r in routData]
            rout = np.array(routData,dtype=dtype)

            print "First 10 a/MAD(a)"
            print mlab.rec2txt(rout[~gRow][:10])
            print "%i there are %i outliers " % (count,goodid[~gRow].size)

            goodid = goodid[gRow]

        count +=1

    return U,S,V,goodid,X2

def mkModes(fdt0,kic0):
    """
    Make Modes

    Take a collection of light curves and find the principle
    components.  This algorithm implements a robust SVD.

    Parameters
    ----------
    fdt : Masked array.  Collection of light curves arranged row-wise.
          Masked elements will be eliminated from the SVD
    kic : Array with the KIC ID
    """

    fdt = fdt0.copy()
    kic = kic0.copy()
    nrow0,ncol0 = fdt.shape

    # Cut out the bad columns and rows.
    # Represent mask as integers, 0=mask.
    # If sum(axis=0)==0, particular cadences in all stars are bad.
    # If sum(axis=1)==0, particular stars are all bad.
    
    mint = (~fdt0.mask).astype(int) # Mask as integers
    bcol = mint.sum(axis=0)!=0 # True - good column
    brow = mint.sum(axis=1)!=0 # True - good row

    nrow = brow[brow].size
    ncol = bcol[bcol].size
    print "%i light curves were entirely masked out" % (brow.size - nrow)
    
    fdt = fdt[:,bcol]
    fdt = fdt[brow,:]
    kic = kic[brow]


    nstars = kic.size    
    fdt[fdt.mask] = 0

    # Normalize by Median Absolute Dev.  Normalized reduced Chi2
    # should be about 1.
    mad = ma.median(ma.abs(fdt),axis=1)
    mad = mad.reshape(mad.size,1)
    fdt = fdt/mad
    fdt = np.vstack(fdt)

    U,S,Vtemp,goodid,X2 = robustSVD(fdt)
    # Cut out the columns that did not pass the robust SVD
    mad = mad[goodid]
    kic = kic[goodid]

    nGood = Vtemp.shape[0] # Number of inlier stars.

    # Fill back in the empty columns
    V = np.zeros((nGood,ncol0))
    V[:,bcol] = Vtemp

    S      = S[:nstars,:nstars]
    A      = np.dot(U,S)
    A      = A[:,:nMode]
    fit    = np.dot(A,V[:nMode])*mad

    return U,S,V,A,fit,kic



def moments(A):
    # Evaluate moments
    names = ['pc','med','mad','mean','max','min']
    dtype = zip(names,[float]*len(names))
    
    rL = np.zeros(A.shape[1],dtype=dtype)
    rL['pc']   = np.arange(A.shape[1])
    rL['mean'] = np.mean(    A , axis=0 )
    rL['med']  = np.median(  A , axis=0 )
    rL['max']  = ma.max(     A , axis=0 )
    rL['min']  = ma.min(     A , axis=0 )
    rL['mad']  = ma.median(  np.abs( A - rL['med'] ) , axis=0 )
    return rL

def cotrendFits(U,S,V,nModes=None):
    """
    Cotrending Fits

    Use the priciple components and singular values, to construct the
    cotrending fits to the light curves
    """

    # Construct fits
    nstars = U.shape[0]
    S      = S[:nstars,:nstars]
    A      = np.dot(U,S)
    A      = A[:,:nModes]
    fits   = np.dot(A,V[:nModes])
    fits   = fit*mad[goodid]
    
    return fits

def calibrate():
    """
    Calibrate Light Curves

    Take a collection of light curves along with the PCA vectors.
    """
def join_on_kic(x1,x2,kic1,kic2):
    """
    Join Arrays on KIC

    Parameters
    ----------

    x1   : First array
    x2   : Second array
    kic1 : KIC ID corresponding to first array
    kic2 : KIC ID corresponding to first array

    Returns
    -------
    xj1  : Joined first array
    xj2  : Joined second array
    kic  : The union of kic1 kic2

    Todo
    ----
    Make default proceedure to just return the indecies like (arg join).

    """
    
    r = lambda kic : rec.fromarrays([kic,np.arange(kic.size)],names='kic,rid')
    r1 = r(kic1)
    r2 = r(kic2)
                                    
    rj = mlab.rec_join('kic',r1,r2)
    kic = rj['kic']

    xj1 = x1[ rj['rid1'] ]
    xj2 = x2[ rj['rid2'] ]
    
    return xj1,xj2,kic


tq = 89.826658388163196
def cutQuarterPeriod(PG):
    """
    Remove multiples of the quarter spacing.
    """
    for i in range(12):
        PG = ma.masked_inside(PG,i*tq-1,i*tq+1)
    for i in range(24):
        PG = ma.masked_inside(PG,i*tq/2-1,i*tq/2+1)
    for i in range(36):
        PG = ma.masked_inside(PG,i*tq/3-1,i*tq/3+1)
    for i in range(48):
        PG = ma.masked_inside(PG,i*tq/4-1,i*tq/4+1)
    return PG

def peaks(mtd,twd):
    mf = nd.maximum_filter(mtd,twd)
    pks = np.unique(mf)
    cnt = np.array([mf[mf==p].size for p in pks])
    pks = pks[cnt==twd]
    return pks

def diag(mtd,twd):
    pks = peaks(mtd,twd)
    pks = sort(pks)

    mad = ma.median(ma.abs(mtd))
    max3day = mean(nd.maximum_filter(mtd,150))

    val   = (pks[-1],mean(pks[-3:]),mean(pks[-10:]),mad  ,max3day)
    dtype = [('maxpk',float),('pk3',float),('pk10',float),('mad',float),('max3day',float)]
    rd = np.array(val,dtype=dtype)

    return rd

def medfit(fdt,vec):
    vec = ma.masked_invalid(vec)
    fdt.mask = vec.mask = fdt.mask | vec.mask
    
    p0 = [0]
    def cost(p):
        return ma.median(ma.abs(fdt-p[0]*vec))
    p1 = optimize.fmin(cost,p0,disp=0)
    fit = ma.masked_array(p1[0] * vec,mask=fdt.mask)
    return fit





twd = 20
#from matplotlib import gridspec
#from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

tprop = dict(size=10,name='monospace')

def compCoTrend(tlc):
    fig = figure(figsize=(18,12))

    algL = ['RawCBV','ClipCBV']
    nalg = len(algL)
    gs = GridSpec(nalg+1,1)

    mtdL = []
    for i in range(nalg):
        alg = algL[i]
        time = tlc.time
        data = ma.masked_array(tlc[alg+'data'],mask=tlc[alg+'datamask'])
        tnd = ma.masked_array(tlc[alg+'tnd'],mask=tlc[alg+'tndmask'])

        pltDiagCoTrend(time,data,tnd,gs=gs[i]) 
        at = AnchoredText(alg,prop=tprop, frameon=True,loc=2)
        gca().add_artist(at)
        
    rcParams['axes.color_cycle'] = ['k','r','c','g']

    axmtd = plt.subplot(gs[-1],sharex=gca())
#    for mtd in mtdL:
#        axmtd.plot(t.TIME,mtd)
#
#    for mtd in mtdL:
#        mf = nd.maximum_filter(mtd,twd)
#        axmtd.plot(t.TIME,mf+50e-6)

#    rdL = [diag(mtd,twd) for mtd in mtdL]
#    rdL = hstack(rdL)
#    for n in rdL.dtype.names:
#        rdL[n] *= 1e6 

#    rdL = mlab.rec_append_fields(rdL,'alg',algL)
#    at = AnchoredText(mlab.rec2txt(rdL,precision=0),prop=tprop, frameon=True,loc=3)

#    axmtd.add_artist(at)
    axL = fig.get_axes()
    for ax in axL:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='both'))
        ax.set_xlim(axL[0].get_xlim())
    ax.xaxis.set_visible(True)

    draw()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001)
    draw()

def pltDiagCoTrend(time,data,fit,gs=None):
    res = data-fit

    if gs is None:
        smgs = GridSpec(4,1)
    else:
        smgs = gridspec.GridSpecFromSubplotSpec(4,1,subplot_spec=gs)

    axres = plt.subplot(smgs[0])
    axres.plot(time,res)

    plt.subplot(smgs[1:],sharex=axres)
    axfit.plot(time,data)
    axfit.plot(time,fit)

def compwrap(tset):
    tset = map(keplerio.ppQ,tset)

    for t in tset:
        compCoTrend(t)
        fig = gcf()
        fig.savefig('%09d.png' % t.keywords['KEPLERID'])


    
    if (alg is 'RawCBV') or (alg is 'ClipCBV') :
        p1 = np.linalg.lstsq( bv[:,bg].T , fm[bg] )[0]
        tnd     = fm.copy()
        tnd[bg] = np.dot( p1 , bv[:,bg] )
        data,tnd = fm,tnd
    elif alg is 'DtSpl':
        data,tnd = fm,fm-fdt
    elif alg is 'DtCBV':
        tndDtCBV = ma.masked_array(tDtCBV.fcbv,mask=fm.mask)
        data,tnd = fdt,tndDtCBV
    elif alg is 'DtMed':
        vec = numpy.load('mom_cycle_q%i.npy' % kw['QUARTER'])
        tndMedCT = medfit(fdt,vec)
        data,tnd  = fdt,tndMedCT



def compDTdata(t):
    """
    Emit the tables used to compare different detrending algorithns.
    """
    tgrid = atpy.Table(masked=True) # Store the different grid search results.
    tlc   = atpy.Table(masked=True) # Store the differnt detrending schemes
    
    tgrid.table_name = 'tgrid'
    tlc.table_name = 'tlc'
    
    fm = ma.masked_array(t.f,mask=t.fmask)

    for alg in ['RawCBV','ClipCBV','DtSpl','DtCBV','DtMed']:
        data,tnd = coTrend(t,alg)
        res = data - tnd
        dM = tfind.mtd(t.TIME,res.data,t.isStep,res.mask,20)

        for name,marr in zip(['data','tnd','dM'],[data,tnd,dM]):
            tlc.add_column(alg+name,marr.data)
            tlc.add_column(alg+name+'mask',marr.mask)
        
        PG,fom = repQper(t,dM,nQ=12)
        tgrid.add_column(alg+'fom',fom)

        
    PG = cutQuarterPeriod(PG)
    tgrid.add_column('PG',PG.data)
    tgrid.add_column('PGmask',PG.mask)
    
    tlc.add_column('time',t.TIME)
        
    tgrid.keywords = t.keywords
    tlc.keywords = t.keywords
    return tgrid,tlc

def repQper(t,dM,nQ=12):
    """
    Extend the quarter and compute the MES 
    """
    
    dM = [dM for i in range(nQ)]
    dM = ma.hstack(dM)

    PG0 = ebls.grid( nQ*90 , 0.5, Pmin=180)
    PcadG,PG = tfind.P2Pcad(PG0)

    res = tfind.pep(t.TIME[0],dM,PcadG)
    return PG,res['fom']


def corr(f1,f2):
    bgood = ~f1.mask & ~f2.mask
    return stats.pearsonr(f1[bgood],f2[bgood])[0]

def mcorr(fdtL):
    nlc = len(fdtL)
    corrmat = zeros((nlc,nlc))

    for i in range(nlc):
        for j in range(nlc):
            corrmat[i,j] = corr(fdtL[i],fdtL[j])

    return corrmat

def indord(ind):
    n = len(ind)
    x,y = np.mgrid[0:n,0:n]
    xs = x[ind]
    return xs

def corrplot(cms,t,fdts,binsize=20):
    """
    Make a correlation plot showing how correlated lightcurves are 
    """
    clf()
    fig = gcf()
    nbins = int(fdts.shape[0] / binsize)
    gs = plt.GridSpec(nbins,3)


    axim = fig.add_subplot(gs[:,0])
    axim.imshow(cms,vmin=0.1,vmax=0.5,aspect='auto',interpolation='nearest')
    axim.set_xlabel("star number")
    axim.set_ylabel("star number")

    for i in range(nbins):
        ax = fig.add_subplot(gs[i,1:3])
        start = i*binsize
        fdtb = [ fdts[i]/ median(fdts[i])  for i in range(start,start+10)]
        fdtb = ma.vstack(fdtb)
        med = ma.median(fdtb,axis=0)
        ax.plot(t,fdtb.T,',')
        ax.plot(t,med,lw=2,color='red')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ylim(-15,15)

    ax.xaxis.set_visible(True)
    tight_layout()
    fig.subplots_adjust(hspace=0.001)
    
def dblock(X,dmi,dma):
    """
    Return block matrix off the diagonal.
    """
    n = X.shape[0]
    nn = dma-dmi
    x,y = mgrid[0:n,0:n]
    return X[(x>=dmi) & (x<dma) & (y>=dmi) & (y<dma)].reshape(nn,nn)

def peaks(mtd,twd):
    mf = nd.maximum_filter(mtd,twd)
    pks = np.unique(mf)
    cnt = np.array([mf[mf==p].size for p in pks])
    pks = pks[cnt==twd]
    return pks

def diag(mtd,twd):
    pks = peaks(mtd,twd)
    pks = sort(pks)

    mad = ma.median(ma.abs(mtd))
    max3day = mean(nd.maximum_filter(mtd,150))

    val   = (pks[-1],mean(pks[-3:]),mean(pks[-10:]),mad  ,max3day)
    dtype = [('maxpk',float),('pk3',float),('pk10',float),('mad',float),('max3day',float)]
    rd = np.array(val,dtype=dtype)

    return rd

def medfit(fdt,vec):
    vec = ma.masked_invalid(vec)
    fdt.mask = vec.mask = fdt.mask | vec.mask
    
    p0 = [0]
    def cost(p):
        return ma.median(ma.abs(fdt-p[0]*vec))
    p1 = optimize.fmin(cost,p0,disp=0)
    fit = ma.masked_array(p1[0] * vec,mask=fdt.mask)
    return fit
