"""
Cotrending code
"""
import glob

from matplotlib import mlab
from matplotlib.gridspec import GridSpec
from matplotlib.pylab import *
from matplotlib.transforms import blended_transform_factory as btf

from scipy import optimize
from scipy import ndimage as nd
from sklearn.decomposition import FastICA, PCA
import pandas as pd
import h5py

import tfind
import detrend
import keplerio
import prepro
import stellar
import h5plus
from plotplus import AddAnchored
from config import nMode,sigOut,maxIt#,path_phot

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

def robustSVD(D,nMode=nMode,sigOut=sigOut,maxIt=maxIt,verbose=True):
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

        if verbose:
            print "Moments of principle component weight"
            print mlab.rec2txt(rL,precision=1)        

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

            if verbose:
                print "First 10 a/MAD(a)"
                print mlab.rec2txt(rout[~gRow][:10],precision=1)
                print "%i there are %i outliers " % (count,goodid[~gRow].size)

            goodid = goodid[gRow]

        count+=1

    return U,S,V,goodid,X2

def robust_components(M0,n_components=4,algo='PCA'):
    """
    Robust Independent Component Analysis

    M : d x n matrix of observations, where d is the dimensionality of
        the measurements, and n is the number of measurements
    """

    M = M0.copy()
    d,n = M0.shape

    # columns of M0, that make it to the end of the sigma-clipping
    icolin = np.arange(n) 
    outliers = -1
    iteration = 0 
    while outliers!=0:
        if algo=='PCA':
            pca = PCA(n_components=n_components)
            U = pca.fit_transform(M)
            V = pca.components_.T
        elif algo=='ICA':
            ica = FastICA(n_components=n_components,max_iter=600)
            U = ica.fit_transform(M)  # Reconstruct signals
            V = ica.mixing_   # Get estimated mixing matrix
        else:
            print "algo = [ICA|PCA]"
            return 

        # Typical dispersion in best fit parameters
        Vsig = 1.48*np.median(abs(V),axis=0) 
        bVin = (abs(V/Vsig) > 5)
        bVin = bVin.sum(axis=1)==0 

        # Throw out outliers
        M = M[:,bVin]
        icolin = icolin[bVin]

        outliers=(~bVin).sum()
        print "Iter %02d: %i outliers" % (iteration,outliers) 

        iteration+=1

    return U,V,icolin

def mkModes(fdt0,kic0,verbose=True,maxIt=maxIt,nMode=nMode):
    """
    Make Modes

    Take a collection of light curves and decompose into principle
    components.  This algorithm implements a robust SVD. We can
    recover the light curves exactly with fdt = dot(A,V)

    Parameters
    ----------

    fdt : Masked array.  Collection of light curves arranged row-wise.
          Masked elements will be eliminated from the SVD
    kic : Array with the KIC ID
    A   : Array of coefficients. Can reconstruct initial array with dot(A,V)
          A[i,j] is coefficient for star i, mode j

    V   : Principle components. Can capture most of the varience via
          dot(A,V[:,:nModes])
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

    U,S,Vtemp,goodid,X2 = robustSVD(fdt,verbose=verbose)
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
    return U,S,V,A,fit,kic,fdt

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


class EnsembleCalibrator:
    """
    Object for performing Ensemble based calibration of light curves.
    """
    def __init__(self):
        pass
    
    def add_training_set(self,fdt,dftr):
        """
        Parameters
        ----------
        fdt : masked array of detrended light curves
              fdt.shape = (N_lightcurves,N_measurements)

        dftr : Pandas DataFrame with an entry for every training light
               curve. Must contain the following keys:
               - epic
               - ra
               - dec
        """
        self.ntr = len(dftr)
        assert fdt.shape[0]==self.ntr,\
            'N_lightcurves must equal stars in catalog'

        # Mask out bad columns and rows
        intmask = (~fdt.mask).astype(int) # Mask as integers
        bcol = intmask.sum(axis=0)!=0 # True - good column
        brow = intmask.sum(axis=1)!=0 # True - good row
        fdt = fdt[:,bcol]

        for n,b in zip(['cadences','light curves'],[bcol,brow]):
            print "%i %s masked out" % (b.size - b.sum(),n)
        print ""

        assert np.alltrue(brow),"cannot handel garbage lightcurves"

        # Fill in the rest of the missing cadences with 0s
        fdt = ma.masked_invalid(fdt)
        fdt.fill_value=0
        fdt.data[:] = fdt.filled()

        # Standardize the light curves
        fdt_sig = 1.48 * median(abs(fdt),axis=1)
        fdt_stand = fdt / fdt_sig[:,newaxis]
        M = fdt_stand.data.T.copy()

        self.fdt = fdt
        self.dftr = dftr
        self.M = M
        self.bcol = bcol
        self.brow = brow

    def robust_components(self, algo='PCA', n_components=8):
        self.algo = algo
        self.n_components = n_components
        self.plot_basename = 'cotrend_robust%s_ncomp=%i' % \
                             (algo,n_components) 

        U,V,icolin = robust_components(self.M, algo=algo, 
                                       n_components=n_components)

        # Output array has bad columns clipped. Resize U to original shape
        U_clip = U.copy()
        U = np.zeros((self.bcol.size,U.shape[1]))
        U[self.bcol,:] = U_clip

        # How many light curves surved iterative clipping?
        nclip = len(icolin)

        A = [bvfitm(self.fdt[i],U_clip.T)[0] for i in icolin]
        A = ma.vstack(A)

        kAs = ['A%02d' % i for i in range(n_components)]

        dftr_clip = self.dftr.iloc[icolin]
        dfAc = pd.DataFrame(A,columns=kAs,index=dftr_clip.index)
        dfA = pd.concat([dftr_clip,dfAc],axis=1)

        # Which stars have inliear coefficients
        dfAc_st = pd.DataFrame(dfAc.median(),columns=['med'])
        dfAc_st['sig'] = 1.48*abs(dfAc - dfAc_st['med']).median()
        dfAc_st['outhi'] = dfAc_st['med'] + 5 * dfAc_st['sig']
        dfAc_st['outlo'] = dfAc_st['med'] - 5 * dfAc_st['sig']
        self.dfAc_st = dfAc_st

        dfA['inlier'] = self.isinlier(dfAc)

        print ""
        print "%i stars in training set " % self.ntr
        print "%i stars survived iterative %s " % (nclip,algo)
        print "%i stars have inlier coefficients" % dfA.inlier.sum()
        
        self.dfA = dfA
        self.kAs = kAs
        self.U = U
        self.V = V

    def isinlier(self,dfAc):
        """
        Returns true if coefficients are in acceptable range
        """
        inlier = (self.dfAc_st.outlo < dfAc) & (dfAc < self.dfAc_st.outhi)
        return np.all(inlier,axis=1)
        
    def bvfitfm(self,fm,mode='ls-map',verbose=True):
        """
        """
        A,fcbv = bvfitm(fm,self.U.T)
        if mode=='ls-map':
            dfAc = pd.DataFrame(pd.Series(A,index=self.kAs)).T
            inlier = self.isinlier(dfAc)[0]
            if not inlier:
                print "outlier not calibrating"
                A[:] = 0
                fcbv[:] =0

        return A,fcbv

    def plot_modes_diag(self, fdt, step=0.001):
        # Compute the fits
        U = self.U
        n_components = U.shape[1]
        ndiag = fdt.shape[0]

        fit = [self.bvfitfm(f)[1] for f in fdt]
        fit = ma.vstack(fit)
        fcal = fdt - fit

        def ses_stats(fstack):
            out = []
            for i in range(fstack.shape[0]):
                df = tfind.ses_stats(fstack[i])
                df['star'] = i
                out+=[df]
            return pd.concat(out,ignore_index=True)
        
        pd.set_option('precision',1)
        def plot_stats(df):
            df = df.groupby('name').median()
            keys = 'rms_1-cad-mean rms_6-cad-mean rms_12-cad-mean'.split()
            s = df[['value']].ix[keys].to_string()
            s = s[s.find('\n')+1:]
            AddAnchored(s,4,prop=dict(family='monospace'))

        df_stats_fdt = ses_stats(fdt)
        df_stats_fcal= ses_stats(fcal)

        # Axis Book-keeping
        fig = figure(figsize=(20,8))

        gs1 = GridSpec(n_components,1)
        gs1.update(left=0.05, right=0.33, wspace=0.05,hspace=0.001)

        ax0 = plt.subplot(gs1[0])

        axPCL = [plt.subplot(gs1[i],sharex=ax0) for i in range(1,n_components)]
        axPCL = [ax0] + axPCL
        setp([ax.get_xaxis() for ax in axPCL[:-1]],visible=False)

        gs2 = GridSpec(1,2)
        gs2.update(left=0.4, right=0.99,hspace=0.1)
        axL = [plt.subplot(gs2[0,i],sharex=ax0) for i in range(2)]
        
        for i in range(n_components):
            sca(axPCL[i])
            plot(U[:,i])

        setp(axPCL[0],title='Top %i %s Components' % (n_components,self.algo) )
        setp([ax for ax in [axPCL[-1]] + axL],xlabel='Measurement Number')
        legend()

        sca(axL[0])
        dy = arange(ndiag)*step

        def plot_label(fstack,**kwargs):
            plot(fstack[0],**kwargs)
            kwargs.pop('label')
            plot(fstack,**kwargs)

        plot_label(fdt.T+dy,color='k',label='Detrended Flux')
        plot_label(fit.T+dy,color='r',label='%s Fit' % self.algo)
        plot_stats(df_stats_fdt)
        legend()

        sca(axL[1])
        plot_label(fcal.T+dy,color='r',label='Residuals')
        plot_stats(df_stats_fcal)
        legend()


# I/O
dsetkeys = 'U'.split()
attrkeys = 'kAs dfAc_st'.split()

def to_hdf(ec,h5file):
    with h5plus.File(h5file) as h5:
        for k in dsetkeys:
            h5[k] = getattr(ec,k)

        r = ec.dfAc_st.to_records()
        rless = mlab.rec_drop_fields(r,['index'])
        sindex = r['index'].astype(str)
        r = mlab.rec_append_fields(rless,'index', sindex)
        h5.attrs['dfAc_st'] = r
        h5.attrs['kAs'] = ec.kAs

def read_hdf(h5file):
    ec = EnsembleCalibrator()
    with h5plus.File(h5file) as h5:
        for k in dsetkeys:
            setattr(ec,k,h5[k][:])

        dfAc_st = pd.DataFrame(h5.attrs['dfAc_st'])
        dfAc_st.index=dfAc_st['index']
        dfAc_st.drop('index',axis=1)
        setattr(ec,'dfAc_st',dfAc_st)
        setattr(ec,'kAs',h5.attrs['kAs'])
    return ec

# Plotting functions
def plot_mode_FOV(dfA,kAs):
    fig,axL = subplots(ncols=4, nrows=2)
    for i,k in zip(range(len(kAs)),kAs):
        sca(axL.flatten()[i])
        scatter(dfA.ra,dfA.dec,c=dfA[k])
        title(k)    
        print dfA[('ra dec %s' % k).split()]

def plot_PCs(U,V):
    fig = figure(figsize=(20,12))
    nPC = U.shape[1]
    gs = GridSpec(nPC,nPC+1)
    axPCL = [fig.add_subplot(gs[i,:-1]) for i in range(nPC)]
    axAL = [fig.add_subplot(gs[i,-1]) for i in range(nPC)]
    for i in range(nPC):
        sca(axPCL[i])
        plot(U[:,i])

        sca(axAL[i])
        hist(V[:,i])

    fig.set_tight_layout(True)

def makeplots(ec,savefig=False):
    dfA = ec.dfA
    kAs = ec.kAs

    plot_PCs(ec.U,ec.V)
    if savefig:
        path = ec.plot_basename+'_PCs.png'
        print "saving %s" % path
        gcf().savefig(path)

    pd.scatter_matrix(dfA[dfA.inlier][kAs],figsize=(8,8))
    if savefig:
        path = ec.plot_basename+'_coeffs.png'
        print "saving %s" % path
        gcf().savefig(path)

    plot_mode_FOV(dfA[dfA.inlier],kAs)
    if savefig:
        gcf().set_tight_layout(True)
        path = ec.plot_basename+'_FOV.png'
        print "saving %s" % path
        gcf().savefig(path)

