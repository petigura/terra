"""
Transit finder: A new approach to finding low signal to noise transits.
"""
from scipy import ndimage as nd
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
import scipy

import numpy as np
from numpy import ma,nan

import ebls

lc = 0.0204343960431288

def LDMF(f,dK):
    """
    Locally-detrended Matched Filter.

    Finds box-shaped signals irrespective of a local continuum slope.


    Parameters
    ----------
    f : flux time series
    dK: Depth kernel.


    Returns
    -------
    dM : Mean depth of transit.

    """
    dM = nd.convolve1d(f,dK,mode='wrap')
    
    return dM


def GenK(twd,fcwd=1):
    """
    Generate Kernels.  
    
    Parameters
    ----------
    twid  : length of the transit (in units of cadence length)
    fcwid : Each side of the continuum is fcwd * twid (default is 1)

    Returns
    -------
    bK : Before transit kernel.
    tK : Trasit kernel.
    aK : After transit kernel.
    dK : Depth kernel.

    Notes
    -----
    Recall that conv(f,g) will flip g.

    """
    cwd = twd * fcwd
    kSize = 2*cwd + twd  # Size of the kernel
    kTemp = np.zeros(kSize) # Template kernal of appropriate length.

    bK = kTemp.copy()
    bK[-cwd:] = np.ones(1) / cwd # Before transit kernel


    boxK = kTemp.copy()
    boxK[cwd:-cwd] = np.ones(1) # Box shaped kernel

    tK = boxK.copy()
    tK[cwd:-cwd] /= twd # transit kernel

    aK = kTemp.copy()
    aK[:cwd] = np.ones(1) / cwd # After transit kernel

    dK = kTemp.copy() 
    dK = 0.5*(bK+aK) - tK # Depth kernel

    return bK,boxK,tK,aK,dK


def dThresh(dM,dMi=50,dMa=1000):
    """
    Depth threshold.

    Applies a cut to the possible transits based on their depth.

    Parameters
    ----------
    dM  : Mean depth of transit.
    dMa : The maximum depth (ppm).
    dMi : The minimum depth (ppm).

    Returns
    -------

    tCand : Boolean array with showing which cadances are cadidate
            mid-transit.

    """
    
    dMi *= 1e-6
    dMa *= 1e-6

    tCand = (dM > dMi) & (dM < dMa)

    return tCand

def pCheck(time,tCand,P0,tbin,nTMi=3):
    """
    Periodicity check.

    Throw out transit cadidates that are not consistent with a single
    period.

    Parameters
    ----------
    time  : Times of each cadance.
    tCand : Boolean array with showing which cadances are cadidate
            mid-transit.
    P0    : Trial period (units of days)
    tbin    : Number of phase bins to check.
    nTMi  : Require that signals repeat a certain number of times.  

    Returns
    -------
    eCand : List of epochs that show 3 transits.

    """
    assert type(tbin) == int

    tbase = np.nanmax(time)-np.nanmin(time)
    epoch = np.arange( tbin,dtype=float ) / tbin * P0
    nT = np.ceil(tbase/P0)
    timeInt = np.arange(tbin*nT)/tbin*P0
    tCandInt = interp1d(time,tCand,kind='nearest',bounds_error=False,
                        fill_value=0)
    
    tCandRS = tCandInt(timeInt).reshape(nT,tbin)
    boolT = tCandRS.sum(axis=0) >= nTMi
    eCand = epoch[np.where(boolT)]

    return eCand

def XWrap(x,ifold,fill_value=0):
    """
    Extend and wrap array.
    
    Fold array every y indecies.  There will typically be a hanging
    part of the array.  This is padded out.

    Parameters
    ----------

    x     : input
    ifold : Wrap array after ifold indecies.

    Return
    ------

    xwrap : Wrapped array.

    """

    assert type(ifold) is int, "ifold must be an int."
    ncad = x.size # Number of cadences
    nrow = int(np.floor(ncad/ifold) + 1)
    nExtend = nrow * ifold - ncad # Pad out remainder of array with 0s.
 
    pad = np.empty(nExtend) 
    pad[:] = fill_value
    x = np.append( x ,pad )
    xwrap = x.reshape( nrow,-1 )

    return xwrap

def lseg(x,P,ph,wid):
    """
    List of segments.

    Fold input according to P.  Return segments of the array at a
    specific phase of a specific width.

    Parameters
    ----------
    x   : Array to be extracted
    P   : Fold array every P indecies (cad)
    ph  : Return every points at ph. (cad)
    wid : Width of each returned segment (cad)

    """
    
    sl = 0 # low end of the slice.
    sh = 0 # Upper end of the slice.
    iseg = 0
    xlseg = []

    n = len(x)
    while sl < n:
        sl = P * iseg + ph - wid/2
        sh = P * iseg + ph + wid/2
        xlseg.append(x[sl:sh])
        iseg += 1

    return xlseg




def boxBR(fsig,dTrans,sig,boxK):
    """
    Box Bayes Ratio.

    Compute the Bayes ratio of Prob( transit ) / Prob( flat ).


    Parameters
    ----------
    fsig   : Detrended flux time series.
    dTrans : Depth of transit.
    boxK   : Model constructed from the same kernel.
    sig    : Point by point sigma.

    Returns
    -------
    BR : log of Bayes Ratio.

    """

    # Compute Chi^2 for the transit model
    tMod = -1.*depth*boxK
    tChi2 = Chi2(fsig,tMod,sig)

    # Compute Chi^2 for the null model
    nlMod = 0.*boxK
    nlChi2 = Chi2(fsig,nlMod,sig)

    BR = tChi2 - nlChi2 

    return BR


def Chi2(data,model,sig):
    """
    Chi^2 with constant sigma.
    """
    
    r = data - model
    nr = r/sig
    chi2 = (nr**2).sum()

    return chi2

def taylorDT(fs,ts,t0,DfDt,f0):
    """
    Taylor series detrending.
    
    Perform a local detrending at a particular cadence.

    Arguments
    ---------
    fs   : Flux segment.
    ts   : Time segment.
    t0   : Time we're expanding about.
    DfDt : Local slope of lightcurve (determined from continuum).
    f0   : Local flux level of continuum.

    Returns
    -------
    trend : Fluxes corresponding to the detrended light curve.

    """
    nptsMi = 10.
    assert fs.size == ts.size, "ts and fs must have equal sizes"
    assert fs.size > nptsMi, "Not enough points"

    trend =  DfDt*(ts - t0) + f0

    return trend


def seg(time,P,ph,wid):
    """
    Segment 

    Return a list of slices corresponding to P,ph and have width = wid

    Arguments
    ---------
    time : numpy array (not masked).
    P    : Period to fold about (days)
    ph   : phase (offset from t = 0)/P
    wid  : Width (days)

    Returns
    -------
    List of slices corresponding to the segment.

    Notes 
    ----- 
    Will not return segments that are shorter than the specified
    width.  If there is missing data, interpolate over it or lose it.
    
    """
    tph = ph*P # time phase.

    tfold = np.mod(time,P)
    tfold = ma.masked_invalid(tfold)
    tfold = ma.masked_outside(tfold,tph-wid/2.,tph+wid/2.)
    sL = ma.notmasked_contiguous(tfold)
    ssL = []
    for s in sL:
        if time[s].ptp() >= wid/2.:
            ssL.append(s)

    return ssL

def FOM(time,fsig,DfDt,f0,P,ep,wid=2,twd=.3,plot=False):
    tfold = np.mod(time,P)

    # Detrend the lightcurve.
    tfold = ma.masked_outside(tfold,ep - wid/2,ep + wid/2)
    tfold = ma.masked_invalid(tfold)

    sLDT = ma.notmasked_contiguous(tfold)
    sLDT = [s for s in sLDT if s.stop-s.start > wid / lc /2]

    fdt = ma.masked_array(fsig,copy=True,mask=True)
    trend = ma.masked_array(fsig,copy=True,mask=True)

    for s in sLDT:
        ms = s.start + wid /lc/2
        trend[s] = taylorDT(fsig[s],time[s],time[ms],DfDt[ms],f0[ms])
        fdt[s] = fsig[s] - trend[s]
 
    # Calculate figure of merit.
    # Detrend the lightcurve.
    tfold.mask = False
    tfold = ma.masked_outside(tfold,ep - twd/2,ep + twd/2)
    tfold = ma.masked_invalid(tfold)
    
    fdt.mask = True
    fdt.mask = tfold.mask
    fdt = ma.masked_invalid(fdt)

    sLFOM = ma.notmasked_contiguous(fdt)
         
    if fdt.count() > 10:
        s2n = -ma.mean(fdt)/ma.std(fdt)*np.sqrt(fdt.count())
    else:
        s2n = 0

    if plot:
        fig = plt.gcf()
        fig.clf()
        ax = plt.gca()

        sDT = sLDT[-1]
        sFOM = sLFOM[-1]

#        ax.plot(time[sDT],fsig[sDT],'.')
#        ax.plot(time[sFOM],fsig[sFOM],'o')
#
#        ax.plot(time[sDT],bM[sDT])
#        ax.plot(time[sDT],aM[sDT])
#        ax.plot(time[sDT],f0[sDT])
#        ax.plot(time[sDT],trend[sDT])

        [ax.plot(tfold.data[s],fsig[s],',k') for s in sLDT] 
        [ax.plot(tfold.data[s],fsig[s],'or',ms=2) for s in sLFOM] 
        [ax.plot(tfold.data[s],trend[s]) for s in sLDT] 
        [ax.axvline(tfold.data[s.start + wid/lc/2]) for s in sLDT] 

        [ax.plot(tfold.data[s],fdt.data[s],'o',alpha=.3) for s in sLDT] 
        [ax.plot(tfold.data[s],fdt.data[s],'o') for s in sLFOM] 

        ax.set_title("epoch = %f" % ep)
        ax.annotate("s2n = %f" % s2n,(.8,.8),xycoords='figure fraction' ,bbox=dict(boxstyle="round", fc="0.8"))

        fig.savefig("test.png" )

    return s2n


def pep(time,fsig,twd,cwd):
    """
    Period-epoch search
    
    Parameters
    ----------
    time - time series
    fsig - flux
    twd - transit width (in days)
    cwd - continuum width (each side, days)
    """

    twd = int(twd/lc) # length of transit in units of cadence
    bK,boxK,tK,aK,dK = GenK( twd ) 
    dM     = nd.convolve1d(fsig,dK)
    bM     = nd.convolve1d(fsig,bK)
    aM     = nd.convolve1d(fsig,aK)
    DfDt   = (aM-bM)/(cwd+twd)/lc
    f0     = 0.5 * (aM + bM)    # Continuum value of fsig (mid transit)




    # Discard cadences that are too high.
    dM = ma.masked_outside(dM,-1e-3,1e-3)
    fsig = ma.masked_array(fsig,mask=dM.mask,fill_value = nan)
    fsig = fsig.filled()

    tCand = dThresh(dM,dMi=100)
    
    # Generate a grid of plausable periods (days)
    PGrid = ebls.grid( np.nanmax(time) - np.nanmin(time) , 0.5, Pmin=50, 
                       Psmp=0.5 )

    eCandGrid = []
    PCandGrid = []

    for P in PGrid:
        eCand = pCheck(time,tCand,P,1000)
        eCandGrid.append( eCand )
        PCand = np.empty( len(eCand) )
        PCand[:] = P
        PCandGrid.append( PCand )


    ee = reduce(np.append,eCandGrid)
    PP = reduce(np.append,PCandGrid)
    print "%i (P,epoch)" % len(ee)

    s2n = np.zeros(len(ee))
    for i in range(len(ee)):
        ep = ee[i]
        P  = PP[i]
        s2n[i] = FOM(time,fsig,DfDt,f0,P,ep,twd=.3,wid=.9)

    return ee,PP,s2n



def pep2(dM,PGrid0,twd,cwd):
    """
    Period-epoch search
    
    Parameters
    ----------
    time - time series
    fsig - flux
    twd - transit width (in days)
    cwd - continuum width (each side, days)
    """

    # The values of period that we sample are different from the input PGrid

#    PGrid = [] 
#    ee = []
#    dd = []
#    for P0 in PGrid0:
#        Pcad = int(round(P0/lc)) # the period in units of cadence
#        PGrid.append( Pcad * lc )
#
#        dMW = XWrap(dM,P0cad,fill_value=np.nan)
#        dMW = ma.masked_invalid(dMW)
#
#        ne,nt = dMW.shape
#        
#
#        epoch = np.arange(
#
#        count = (~dMW.mask).astype(int).sum(axis=0)
#        bcount = count >= 3
    


    # Discard cadences that are too high.
    dM = ma.masked_outside(dM,-1e-3,1e-3)
    fsig = ma.masked_array(fsig,mask=dM.mask,fill_value = nan)
    fsig = fsig.filled()

    tCand = dThresh(dM,dMi=100)
    
    # Generate a grid of plausable periods (days)
    PGrid = ebls.grid( np.nanmax(time) - np.nanmin(time) , 0.5, Pmin=50, 
                       Psmp=0.5 )

    eCandGrid = []
    PCandGrid = []

    for P in PGrid:
        eCand = pCheck(time,tCand,P,1000)
        eCandGrid.append( eCand )
        PCand = np.empty( len(eCand) )
        PCand[:] = P
        PCandGrid.append( PCand )


    ee = reduce(np.append,eCandGrid)
    PP = reduce(np.append,PCandGrid)
    print "%i (P,epoch)" % len(ee)

    s2n = np.zeros(len(ee))
    for i in range(len(ee)):
        ep = ee[i]
        P  = PP[i]
        s2n[i] = FOM(time,fsig,DfDt,f0,P,ep,twd=.3,wid=.9)

    return ee,PP,s2n



def tdpep(time,fsig):
    """
    Transit-duration - Period - Epoch
    """

    tdur = np.linspace(0.3,0.8,4)
    tt = []
    PP = []
    ee = []
    s2n = []

    for t in tdur:
        e,P,s = pep(time,fsig,t,t)
        PP.append(P)
        ee.append(e)
        s2n.append(s)
        tt.append(np.zeros(len(P))+t)
        
    tt = reduce(np.append,tt)
    PP = reduce(np.append,PP)
    ee = reduce(np.append,ee)
    s2n = reduce(np.append,s2n)

    return tt,PP,ee,s2n




def cadFill(cad0):
    """
    Cadence Fill

    We want the elements of the arrays to be evenly sampled so that
    phase folding is equivalent to array reshaping.

    Parameters
    ----------
    cad - Array of cadence identifiers.
    
    Returns
    -------
    cad   - New array of cadences (without gaps).
    iFill - Indecies that were not missing.

    """
    
    bins = np.arange(cad0[0],cad0[-1]+2)
    count,cad = np.histogram(cad0,bins=bins)
    iFill = np.where(count == 1)[0]
    
    return cad,iFill
