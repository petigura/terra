"""
Transit Validation

After the brute force period search yeilds candidate periods,
functions in this module will check for transit-like signature.
"""
from scipy.spatial import cKDTree
import sqlite3
import h5plus
import re
import copy

import numpy as np
from numpy import ma
import numpy.random as rand 
from scipy import optimize
from scipy import ndimage as nd
from scipy.stats import ks_2samp

import FFA
import keptoy
import tfind
from matplotlib import mlab
import h5py
import os
from matplotlib.cbook import is_string_like,is_numlike
import keplerio
import config
from emcee import EnsembleSampler
import pandas as pd

def scar(res):
    """
    Determine whether points in p,epoch plane are scarred.  That is
    high MES values are due to single point outliers.

    Parameters
    ----------
    res  : result array with `Pcad`, `t0cad`, and `s2n` fields

    Returns
    -------
    nn   : 90 percentile distance to nearest neighbor.  
    
    """

    
    bcut = res['s2n']> np.percentile(res['s2n'],90)
    x = res['Pcad'][bcut]
    x -= min(x)
    x /= max(x)
    y = (res['t0cad']/res['Pcad'])[bcut]

    D = np.vstack([x,y]).T
    tree = cKDTree(D)
    d,i= tree.query(D,k=2)
    return np.percentile(d[:,1],90)

def t0shft(t,P,t0):
    """
    Epoch shift

    Find the constant shift in the timeseries such that t0 = 0.

    Parameters
    ----------
    t  : time series
    P  : Period of transit
    t0 : Epoch of one transit (does not need to be the first).

    Returns
    -------
    dt : Amount to shift by.
    """
    t  = t.copy()
    dt = 0

    t  -= t0 # Shifts the timeseries s.t. transits are at 0,P,2P ...
    dt -= t0

    # The first transit is at t =  nFirstTransit * P
    nFirstTrans = np.ceil(t[0]/P) 
    dt -= nFirstTrans*P 

    return dt

def transLabel(t,P,t0,tdur,cfrac=1,cpad=0):
    """
    Transit Label

    Mark cadences as:
    - transit   : in transit
    - continuum : just outside of transit (used for fitting)
    - other     : all other data

    Parameters
    ----------
    t     : time series
    P     : Period of transit
    t0    : epoch of one transit
    tdur  : transit duration
    cfrac : continuum defined as points between tdur * (0.5 + cpad)
            and tdur * (0.5 + cpad + cfrac) of transit midpoint cpad
    cpad  : how far away from the transit do we start the continuum
            region in units of tdur.


    Returns
    -------

    A record array the same length has the input. Most of the indecies
    are set to -1 as uninteresting regions. If a region is interesting
    (transit or continuum) I label it with the number of the transit
    (starting at 0).

    - tLbl      : Index closest to the center of the transit
    - tRegLbl   : Transit region
    - cRegLbl   : Continuum region
    - totRegLbl : Continuum region and everything inside

    Notes
    -----
    tLbl might not be the best way to find the mid transit index.  In
    many cases, dM[rec['tLbl']] will decrease with time, meaning there
    is a cumulative error that's building up.

    """

    t = t.copy()
    t += t0shft(t,P,t0)

    names = ['totRegLbl','tRegLbl','cRegLbl','tLbl']
    rec  = np.zeros(t.size,dtype=zip(names,[int]*len(names)) )
    for n in rec.dtype.names:
        rec[n] -= 1
    
    iTrans   = 0 # number of transit, starting at 0.
    tmdTrans = 0 # time of iTrans mid transit time.  
    while tmdTrans < t[-1]:
        # Time since mid transit in units of tdur
        t0dt = np.abs(t - tmdTrans) / tdur 
        it   = t0dt.argmin()
        bt   = t0dt < 0.5
        bc   = (t0dt > 0.5 + cpad) & (t0dt < 0.5 + cpad + cfrac)
        btot = t0dt < 0.5 + cpad + cfrac
        
        rec['tRegLbl'][bt] = iTrans
        rec['cRegLbl'][bc] = iTrans
        rec['tLbl'][it]    = iTrans
        rec['totRegLbl'][btot] = iTrans

        iTrans += 1 
        tmdTrans = iTrans * P

    return rec

def LDT(t,fm,recLbl,verbose=False,deg=1,nCont=4):
    """
    Local Detrending

    A simple function that subtracts a polynomial trend from the
    continuum region on either side of a transit.  There must be at
    least two valid data points on either side of the transit to
    include it in the fit.

    Parameters
    ----------
    t      : time
    fm     : masked flux array
    recLbl : record array with following labels
             - cRegLbl   : Region to fit the continuum
             - totRegLbl : Entire region to detrend.
             - tLbl   : Middle of transit

    nCont  : Number of continuum points before and after transit in
             order to use the transit

    Returns
    -------
    fldt : local detrended flux
    """
    fldt    = ma.masked_array( np.zeros(t.size) , True ) 
    assert type(fm) is np.ma.core.MaskedArray,'Must have mask'
    cLbl = recLbl['cRegLbl']
    for i in range(cLbl.max() + 1):
        # Points corresponding to continuum region.
        bc = (cLbl == i ) 

        fc = fm[bc]  # masked array
        tc = t[bc]

        # Identifying the detrending region.
        bldt = (recLbl['totRegLbl']==i)
        tldt = t[bldt]

        # Times of transit
        bt = (recLbl['tLbl'] == i )
        tt = t[bt]

        # There must be a critical number of valid points before and
        # after the transit.
        ncBefore = fc[ tc < tt ].count()  
        ncAfter  = fc[ tc > tt ].count()  

        if (ncBefore >= nCont) & (ncAfter >= nCont) :        
            tc   -= tt
            tldt -= tt

            tc = tc[~fc.mask]
            fc = fc[~fc.mask]

            pfit = np.polyfit(tc,fc,deg)
            fcontfit = np.polyval(pfit,tldt)

            fldt[bldt] = fm[bldt] - fcontfit
            fldt.mask[bldt] = fm.mask[bldt]
        elif verbose==True:
            print "no data for trans # %i" % (i)
        else:
            pass
        
    return fldt

def PF(t,fm,P,t0,tdur,cfrac=3,cpad=1,LDT_deg=1,nCont=4):
    """
    Phase Fold
    
    Convience function that runs the local detrender and then phase
    folds the lightcurve.

    Parameters
    ----------

    t    : time
    fm   : masked flux array
    P    : Period (days)
    t0   : epoch (days)
    tdur : transit width (days)


    cfrac, cpad : passed to transLabel
    deg         : passed to LDT
    deg  : degree of the polynomial fitter

    Returns
    -------
    tPF  : Phase folded time.  0 corresponds to mid transit.
    fPF  : Phase folded flux.
    """
    recLbl = transLabel(t,P,t0,tdur,cfrac=cfrac,cpad=cpad)
    fldt = LDT(t,fm,recLbl,deg=LDT_deg,nCont=nCont) # full size array
    tm = ma.masked_array(t,fldt.mask,copy=True)
    dt = t0shft(t,P,t0)
    tm += dt

    dtype = zip(['tPF','f','t'],[float]*3)
    cut   = ~tm.mask
    lcPF  = np.array( zip(tm[cut],fldt[cut],t[cut]) ,  dtype=dtype )

    lcPF['tPF'] = np.mod(lcPF['tPF']+P/2,P)-P/2
    lcPF = lcPF[np.argsort(lcPF['tPF'])]

    bg  = ~np.isnan(lcPF['tPF']) & ~np.isnan(lcPF['f'])
    lcPF = lcPF[bg]    
    lcPF['tPF'] += config.lc
    return lcPF

# Must add the following attributes
# 't0','Pcad','twd','mean','s2n','noise'
# tdur
# climb

def read_dv(h5,tpar=False):
    """
    Read h5 file for data validation

    1. Finds the highest SNR peak
    2. Adds oft-used variables to top level

    Parameters
    ----------
    h5   : h5 file (post grid-search)
    tpar : alternative transit ephemeris dictionary with the following keys
           t0 Pcad twd mean s2n noise
    """
    h5.lc  = h5['/pp/mqcal'][:] # Convenience
    h5.RES = h5['/it0/RES'][:]

    id  = np.argmax(h5.RES['s2n'])        
    for k in 't0 Pcad twd mean s2n noise'.split():
        if tpar==False:
            h5.attrs[k] = h5.RES[k][id]
        else:
            h5.attrs[k] = tpar[k] 

    h5.attrs['tdur'] = h5.attrs['twd']*config.lc
    h5.attrs['P']    = h5.attrs['Pcad']*config.lc

    # Convenience
    h5.fm = ma.masked_array(h5.lc['fcal'],h5.lc['fmask'])
    h5.dM = tfind.mtd(h5.lc['t'],h5.fm,h5.attrs['twd'])
    h5.t  = h5.lc['t']

def checkHarmh5(h5):
    """
    If one of the harmonics has higher MES, update P,epoch
    """
    P = h5.attrs['P']
    harmL,MESL = checkHarm(h5.dM,P,h5.t.ptp(),ver=False)
    idma = np.nanargmax(MESL)
    if idma != 0:
        harm = harmL[idma]
        print "%s P has higher MES" % harm
        h5.attrs['P']    *= harm
        h5.attrs['Pcad'] *= harm

def at_phaseFold(h5,ph):
    """ 
    Add locally detrended light curve

    Parameters
    ----------

    h5 - h5 file with the following columns
         mqcal
    """

    attrs = h5.attrs
    
    lc = h5['/pp/mqcal'][:]
    t  = lc['t']
    fm = ma.masked_array(lc['f'],lc['fmask'])

    keys  = ['LDT_deg','cfrac','cpad','nCont']
    kw    = {}
    for k in keys:
        kw[k] = attrs[k]

    P,tdur = attrs['P'],attrs['tdur']
    t0     = attrs['t0'] + ph / 360. * P 

    rPF   = PF(t,fm,P,t0,tdur,**kw)
    qarr = keplerio.t2q( rPF['t'] ).astype(int)
    rPF  = mlab.rec_append_fields(rPF,'qarr',qarr)
    h5['lcPF%i' % ph] = rPF

def at_binPhaseFold(h5,ph,bwmin):
    """
    Add the binned quantities to the LC

    h5 - h5 file with the following columns
         lcPF0
         lcPF180         
    ph - phase to fold at.
    bwmin - targe bin width (minutes)
    """
    lcPF = h5['lcPF%i' % ph][:]
    tPF  = lcPF['tPF']
    f    = lcPF['f']

    xmi,xma = tPF.min(),tPF.max() 
    bw       = bwmin / 60./24. # converting minutes to days
    nbins    = xma-xmi

    nbins = int( np.round( (xma-xmi)/bw ) )
    # Add a tiny bit to xma to get the last element
    bins  = np.linspace(xmi,xma+bw*0.001,nbins+1 )
    tb    = 0.5*(bins[1:]+bins[:-1])

    blcPF =  np.array([tb],dtype=[('tb',float)])
    dfuncs = {'count':ma.count,'mean':ma.mean,'med':ma.median,'std':ma.std}
    for k in dfuncs.keys():
        func  = dfuncs[k]
        yapply = bapply(tPF,f,bins,func)
        blcPF = mlab.rec_append_fields(blcPF,k,yapply)

    blcPF = blcPF.flatten()
    
    d = dict(ph=ph,bwmin=bwmin)
    name  = 'blc%(bwmin)iPF%(ph)i' % d
    
    h5[ name ] = blcPF
    for k in d.keys():
        h5[ name ].attrs[k] = d[k]

def at_fit(h5,runmcmc=True):
    """
    Attach Fit

    Fit Mandel Agol (2002) to phase folded light curve. For short
    period transits with many in-transit points, we fit median phase
    folded flux binned up into 10-min bins to reduce computation time.

    Light curves are fit using the following proceedure:
    1. Registraion : best fit transit epcoh is determined by searching
                     over a grid of transit epochs
    2. Simplex Fit : transit epoch is frozen, and we use a simplex
                     routine (fmin) to search over the remaining
                     parameters.
    3. MCMC        : If runmcmc is set, explore the posterior parameter space
                     around the best fit results of 2.

    Parameters
    ----------
    h5     : h5 file handle

    Returns
    -------
    Modified h5 file. Add/update the following datasets in h5['fit/']
    group:

    fit/t     : time used in fit
    fit/f     : flux used in fit
    fit/fit   : best fit light curve determined by simplex algorithm
    fit/chain : values used in MCMC chain post burn-in

    And the corresponding attributes
    fit.attrs['pL0']  : best fit parameters from simplex alg
    fit.attrs['X2_0'] : chi2 of fit
    fit.attrs['dt_0'] : Shift used in the regsitration.
    """

    attrs = dict(h5.attrs)
    trans = TM_read_h5(h5)
    trans.register()
    trans.pdict = trans.fit()[0]
    if runmcmc:
        trans.MCMC()
    TM_to_h5(trans,h5)

def at_med_filt(h5):
    """Add median detrended lc"""
    lc = h5['/pp/mqcal']
    fm = ma.masked_array(lc['fcal'],lc['fmask'])

    t  = lc['t']
    P  = h5.attrs['P']
    t0 = h5.attrs['t0']
    tdur= h5.attrs['tdur']
    twd = h5.attrs['twd']
    y = h5.fm-nd.median_filter(h5.fm,size=twd*3)
    h5['fmed'] = y

    # Shift t-series so first transit is at t = 0 
    dt = t0shft(t,P,t0)
    tf = t + dt
    phase = np.mod(tf+P/4,P)/P-1./4
    x = phase * P
    h5['tPF'] = x

    # bin up the points
    for nbpt in [1,5]:
        bins = get_bins(h5,x,nbpt)
        y = ma.masked_invalid(y)
        bx,by = hbinavg(x[~y.mask],y[~y.mask],bins)
        h5['bx%i'%nbpt] = bx
        h5['by%i'%nbpt] = by

def get_bins(h5,x,nbpt):
    """Return bins of a so that nbpt fit in a transit"""
    nbins = np.round( x.ptp()/h5.attrs['tdur']*nbpt ) 
    return np.linspace(x.min(),x.max(),nbins+1)


def at_s2ncut(h5):
    """
    Cut out the transit and recomput S/N.  It s2ncut should be low.
    """
    attrs = h5.attrs

    lbl = transLabel(h5.t,attrs['P'],attrs['t0'],attrs['tdur'])

    # Notch out the transit and recompute
    fmcut = h5.fm.copy()
    fmcut.mask = fmcut.mask | (lbl['tRegLbl'] >= 0)
    dMCut = tfind.mtd(h5.t,fmcut, attrs['twd'] )    

    Pcad0 = np.floor(attrs['Pcad'])
    r = tfind.ep(dMCut, Pcad0)

    i = np.nanargmax(r['mean'])
    if i is np.nan:
        s2n = np.nan
    else:
        s2n = r['mean'][i]/attrs['noise']*np.sqrt(r['count'][i])

    h5.attrs['s2ncut']      = s2n
    h5.attrs['s2ncut_t0']   = r['t0cad'][i]*keptoy.lc + h5.t[0]
    h5.attrs['s2ncut_mean'] = r['mean'][i]

def at_SES(h5):
    """
    Look at individual transits SES.
    
    Finds the cadence closes to the transit midpoint. For each
    cadence, record in dataset `SES` the following information:

        ses    : Depth of the transit
        tnum   : transit number (starts at 0)
        season : What season did the transit occur in?

    Also computes the median for odd/even transits and each season
    individually.

    Notes
    -----
    Aug 5, 2013 - Changed to strictly look at the point closest to the
                  center of each transit. Before, I was searching for
                  the max SES value within the transit range (allowed
                  for TTVs)
    """
    lc   = h5['/pp/mqcal'][:]
    t    = lc['t']
    fcal = ma.masked_array(lc['fcal'],lc['fmask'])
    dM   = h5.dM

    attrs = h5.attrs
    rLbl = transLabel(t,attrs['P'],attrs['t0'],attrs['tdur']*2)
    qrec = keplerio.qStartStop()
    q = np.zeros(t.size) - 1
    for r in qrec:
        b = (t > r['tstart']) & (t < r['tstop'])
        q[b] = r['q']

    season = np.mod(q,4)
    tRegLbl = rLbl['tRegLbl']

    dtype = [('ses',float),('tnum',int),('season',int)]
    dM.fill_value=np.nan
    rses  = np.array(zip(dM.filled(),rLbl['tLbl'],season),dtype=dtype )
    rses  = rses[rLbl['tLbl'] >=0]

    h5.attrs['num_trans'] = 0

    if rses.size > 0:
        h5['SES'] = rses
        h5.attrs['num_trans'] = rses.size
        h5['tRegLbl'] = tRegLbl

        # Median SES, even/odd
        for sfx,i in zip(['even','odd'],[0,1]):
            medses =  np.median( rses['ses'][rses['tnum'] % 2 == i] ) 
            h5.attrs['SES_%s' %sfx] = medses

        # Median SES, different seasons
        for i in range(4):
            medses = np.median( rses['ses'][rses['season'] % 4  == i] ) 
            h5.attrs['SES_%i' %i] = medses

def at_rSNR(h5):
    """
    Robust Signal to Noise Ratio
    """
    ses = h5['SES'][:]['ses'].copy()
    ses.sort()
    h5.attrs['clipSNR'] = np.mean(ses[:-3]) / h5.attrs['noise'] *np.sqrt(ses.size)
    x = np.median(ses) 
    h5.attrs['medSNR'] =  np.median(ses) / h5.attrs['noise'] *np.sqrt(ses.size)


def at_s2n_known(h5,d):
    """
    When running a simulation, we know a priori where the transit
    will fall.  This function attaches the s2n_known given the
    closest P,t0,twd
    """                
    tup = s2n_known(d,h5.t,h5.fm)

    h5.attrs['twd_close']   = tup[2]
    h5.attrs['P_close']     = tup[3]
    h5.attrs['phase_close'] = tup[4]
    h5.attrs['s2n_close']   = tup[5]


def at_autocorr(h5):
    bx5fft = np.fft.fft(h5['by5'][:].flatten() )
    corr = np.fft.ifft( bx5fft*bx5fft.conj()  )
    corr = corr.real
    lag  = np.fft.fftfreq(corr.size,1./corr.size)
    lag  = np.fft.fftshift(lag)
    corr = np.fft.fftshift(corr)
    b = np.abs(lag > 6) # Bigger than the size of the fit.

    h5['lag'] = lag
    h5['corr'] = corr
    h5.attrs['autor'] = max(corr[~b])/max(np.abs(corr[b]))

def at_grass(h5):
    """
    Start with the tallest SNR period. Compute the median height of
    three nearby peaks?
    """

    P   = h5.attrs['P']
    fac = 1.4 # bin ranges from P/fac to P*fac.
    bins  = np.logspace(np.log10(P/fac),np.log10(P*fac),101)
    xp,yp  =findpks(h5.RES['Pcad']*config.lc,h5.RES['s2n'],bins)

    h5.attrs['grass'] = np.median(np.sort(yp)[-5:]) # enough to ignore
                                                    # the primary peak

def findpks(x,y,bins):
    id    = np.digitize(x,bins)
    uid   = np.unique(id)[1:-1] # only elements inside the bin range

    mL    = [np.max(y[id==i]) for i in uid]
    mid   = [ np.where((y==m) & (id==i))[0][0] for i,m in zip(uid,mL) ] 
    return x[mid],y[mid]

######
# IO #
######

def write(h5,pkfile,**kwargs):
    hpk = h5plus.File(pkfile,**kwargs)

    for k in h5.attrs.keys():
        hpk.attrs[k] = h5.attrs[k]
    for k in h5.keys():
        hpk.create_dataset(k,data=h5[k])
    hpk.close()

def flatten(h5,exclRE):
    """
    Return a flat dictionary with exclRE keys excluded.
    """
    pkeys = h5.attrs.keys() 
    pkeys = [k for k in pkeys if re.match(exclRE,k) is None]
    pkeys.sort()

    d = {}
    for k in pkeys:
        v = h5.attrs[k]
        if k.find('pL') != -1:
            suffix = k.split('pL')[-1]
            d['df'+suffix]  = v[0]**2
            d['tau'+suffix] = v[1]
            d['b'+suffix]   = v[2]
        else:
            d[k] = v
    return d

def pL2d(h5,pL):
    return dict(df=pL[0],tau=pL[1],b=pL[2])

def diag_leg(h5):
    dprint = flatten(h5,h5.noDiagRE)
    return dict2str(h5,dprint)

def dict2str(h5,dprint):
    """
    """

    strout = \
"""
%s
-------------
""" % h5.attrs['skic']

    keys = dprint.keys()
    keys.sort()
    for k in keys:
        v = dprint[k]
        if is_numlike(v):
            if v<1e-3:
                vstr = '%.4g [ppm] \n' % (v*1e6)
            else:
                vstr = '%.4g \n' % v
        elif is_string_like(v):
            vstr = v+'\n'

        strout += k.ljust(12) + vstr

    return strout

def get_db(h5):
    """
    Return a dict to store as db
    """
    d = h5.flatten(h5.noDBRE)

    dtype = []
    for k in d.keys():
        k = str(k) # no unicode
        v = d[k]
        if is_string_like(v):
            typ = '|S100'
        elif is_numlike(v):
            typ = '<f8'
        dtype += [(k,typ)]

    t = tuple(d.values())
    return np.array([t,],dtype=dtype)

def bapply(x,y,bins,func):
    """
    Apply an accumulating function on a bin by bin basis.
    
    Parameters
    ----------
    x    : independent var
    y    : dependent var
    bins : passed to digitize
    func : must take an array as input and return a single value
    """
    
    assert bins[0]  <= min(x),'range'
    assert bins[-1] >  max(x),'range'

    bid = np.digitize(x,bins)    
    nbins   = bins.size-1
    yapply  = np.zeros(nbins)

    for id in range(1,nbins):
        yb = y[bid==id]
        yapply[id-1] = func(yb)

    return yapply


def hbinavg(x,y,bins):
    """
    Computes the average value of y on a bin by bin basis.

    x - array of x values
    y - array 
    bins - array of bins in x
    """

    binx = bins[:-1] + (bins[1:] - bins[:-1])/2.
    bsum = ( np.histogram(x,bins=bins,weights=y) )[0]
    bn   = ( np.histogram(x,bins=bins) )[0]
    biny = bsum/bn

    return binx,biny

def s2n_known(d,t,fm):
    """
    Compute the FFA S/N assuming we knew where the signal was before hand

    Parameters
    ----------

    d  : Simulation Parameters
    t  : time
    fm : lightcurve (maksed)

    """
    # Compute the twd from the twdG with the closest to the injected tdur
    tdur       = keptoy.tdurMA(d)
    itwd_close = np.argmin(np.abs(np.array(config.twdG) - tdur/config.lc))
    twd_close  = config.twdG[itwd_close]
    dM         = tfind.mtd(t,fm,twd_close)

    # Fold the LC on the closest period we're computing all the FFA
    # periods around the P closest.  This is a little complicated, but
    # it allows us to use the same functions as we do in the complete
    # period scan.

    Pcad0 = int(np.floor(d['P'] / config.lc))
    t0cad,Pcad,meanF,countF = tfind.fold(dM,Pcad0)

    iPcad_close  = np.argmin(np.abs(Pcad - d['P']/config.lc  ))
    Pcad_close   = Pcad[iPcad_close]
    P_close      = Pcad_close * config.lc
    
    # Find the epoch that is closest to the injected phase. 
    t0    = t[0]+ t0cad * config.lc # Epochs in days.
    phase = np.mod(t0,P_close)/P_close    # Phases [0,1)

    # The following 3 lines compute the difference between the FFA
    # phases and the injected phase.  Phases can be measured going
    # clockwise or counter clockwise, so we must choose the minmum
    # value.  No phases are more distant than 0.5.
    dP = np.abs(phase-d['phase']) 
    dP = np.vstack([dP,1-dP])
    dP = dP.min(axis=0)      

    iphase_close = np.argmin(dP)
    phase_close  = phase[iphase_close]

    # s2n for the closest twd,P,phas
    noise = ma.median( ma.abs(dM) )   
    s2nF = meanF / noise *np.sqrt(countF) 
    s2nP = s2nF[iPcad_close] # length Pcad0 array with s2n for all the
                             # P at P_closest
    s2n_close =  s2nP[iphase_close]    

    return phase,s2nP,twd_close,P_close,phase_close,s2n_close

def checkHarm(dM,P,tbase,harmL0=config.harmL,ver=True):
    """
    Evaluate S/N for (sub)/harmonics
    """
    MESL  = []
    Pcad  = P / config.lc

    dMW = FFA.XWrap(dM , Pcad)
    harmL = []
    for harm in harmL0:
        # Fold dM on the harmonic
        Pharm = P*float(harm)
        if Pharm < tbase / 3.:
            dMW_harm = FFA.XWrap(dM,Pharm / config.lc )
            sig  = dMW_harm.mean(axis=0)
            c    = dMW_harm.count(axis=0)
            MES = sig * np.sqrt(c) 
            MESL.append(MES.max())
            harmL.append(harm)
    MESL = np.array(MESL)
    MESL /= MESL[0]

    if ver:
        print harmL
        print MESL
    return harmL,MESL

def nT(t,mask,p):
    """
    Simple helper function.  Given the transit ephemeris, how many
    transit do I expect in my data?
    """
    trng = np.floor( 
        np.array([p['epoch'] - t[0],
                  t[-1] - p['epoch']]
                 ) / p['P']).astype(int)

    nt = np.arange(trng[0],trng[1]+1)
    tbool = np.zeros(nt.size).astype(bool)
    for i in range(nt.size):
        it = nt[i]
        tt = p['epoch'] + it*p['P']
        # Find closest time.
        dt = abs(t - tt)
        imin = np.argmin(dt)
        if (dt[imin] < 0.3) & ~mask[imin]:
            tbool[i] = True

    return tbool

class TransitModel:
    """
    TransitModel
    
    Simple class for fitting transit    
    """
    def __init__(self,t,f,ferr,climb,pdict,
                 fixdict=dict(p=False,tau=False,b=False,dt=True)):
        """
        Constructor for transit model.

        Parameters
        ----------
        t     : time array
        f     : flux array
        ferr  : errors
        climb : size 4 array with non-linear limb-darkening components
        pL    : Parameter list dictionary [ Rp/Rstar, tau, b, dt]. 
        fix   : Same keys as above, determines which parameters to fix
        """
        # Protect input arrays 
        self.t       = t.copy()
        self.f       = f.copy()
        self.ferr    = ferr.copy()
        self.climb   = climb
        self.pdict   = pdict
        self.fixdict = fixdict

        # Strip nans from t, y, and err
        b = np.vstack( map(np.isnan, [t,f,ferr]) ) 
        b = b.astype(int).sum(axis=0) == 0 
        b = b & (ferr > 0.)

        if b.sum() < b.size:
            print "removing %i measurements " % (b.size - b.sum())
            self.t    = t[b]
            self.f    = f[b]
            self.ferr = ferr[b]

    def register(self):
        """
        Register light curve
        
        Transit center will usually be within 1 long cadence
        measurement. We search for the best t0 over a finely sampled
        grid that spans twice the transit duration. At each trial t0,
        we find the best fitting MA model. We adopt the displacement
        with the minimum overall Chi2.
        """
        # For the starting value to the optimization, use the current
        # parameter vector.
        pdict0 = copy.copy(self.pdict)
        tau0   = pdict0['tau']

        dt_per_lc = 3   # Number trial t0 per grid point
        dtarr = np.linspace(-2*tau0,2*tau0,4*tau0/keptoy.lc*dt_per_lc)

        def f(dt):
            self.pdict['dt'] = dt
            pdict,X2 = self.fit()
            return X2

        X2L = np.array( map(f,dtarr) )
        if (~np.isfinite(X2L)).sum() != 0:
            print "problem with registration: setting dt to 0.0"
            self.pdict['dt'] = 0.
        else:
            self.pdict['dt'] = dtarr[np.argmin(X2L)]
            print "registration: setting dt to %(dt).2f" % self.pdict

    def fit(self):
        """
        Fit MA model to LC

        Run the Nelder-Meade minimization routine to find the
        best-fitting MA light curve. If we are holding impact
        parameter fixed.
        """
        pL   = self.pdict2pL(self.pdict)
        res  = optimize.fmin(self.chi2,pL,disp=False,full_output=True)
        pdict = self.pL2pdict(res[0])
        X2    = res[1]
        return pdict,X2

    def MCMC(self):
        """
        Run MCMC

        Explore the parameter space around the current pL with MCMC

        Adds the following attributes to the transit model structure:

        upL0  : 3x3 array of the 15, 50, and 85-th percentiles of
                Rp/Rstar, tau, and b
        chain : nsamp x 3 array of the parameters tried in the MCMC chain.
        fits  : Light curve fits selected randomly from the MCMC chain.
        """
        
        # MCMC parameters
        nwalkers = 10; ndims = 3
        nburn   = 1000
        niter   = 2000
        print """\
running MCMC
------------
%6i walkers
%6i step burn in
%6i step run
""" % (nwalkers,nburn,niter)

        # Initialize walkers
        pL  = self.pdict2pL(self.pdict)
        fltpars = [ k for k in self.pdict.keys() if not self.fixdict[k] ]
        allpars = self.pdict.keys()
        p0  = np.vstack([pL]*nwalkers) 
        for i,name in zip(range(ndims),fltpars):
            if name=='p':
                p0[:,i] += 1e-4*rand.randn(nwalkers)
            elif name=='tau':
                p0[:,i] += 1e-2*pL[i]*rand.random(nwalkers)
            elif name=='b':
                p0[:,i] = 0.8*rand.random(nwalkers) + .1

        # Burn in 
        sampler = EnsembleSampler(nwalkers,ndims,self)
        pos, prob, state = sampler.run_mcmc(p0, nburn)

        # Real run
        sampler.reset()
        foo   = sampler.run_mcmc(pos, niter, rstate0=state)

        chain  = pd.DataFrame(sampler.flatchain,columns=fltpars)
        uncert = pd.DataFrame(index=['15,50,85'.split(',')],columns=allpars)
        for k in self.pdict.keys():
            if self.fixdict[k]:
                chain[k]  = self.pdict[k]
                uncert[k] = self.pdict[k]
            else:
                uncert[k] = np.percentile( chain[k], [15,50,85] )

        self.chain  = chain.to_records(index=False)
        self.uncert = uncert.to_records(index=False)

        nsamp = 200
        ntrial = sampler.flatchain.shape[0]
        id = np.random.random_integers(0,ntrial-1,nsamp)

        f = lambda i : self.MA( self.pL2pdict(sampler.flatchain[i]),self.t)
        self.fits = np.vstack( map(f,id) )         

    def __call__(self,pL):
        """
        Used for emcee MCMC routine
        
        pL : list of parameters used in the current MCMC trial
        """
        loglike = -self.chi2(pL)
        return loglike


    def chi2(self,pL):
        pdict   = self.pL2pdict(pL)
        f_model = self.MA(pdict,self.t)
        resid   = (self.f - f_model)/self.ferr
        X2 = (resid**2).sum()
        if (pdict['tau'] < 0) or (pdict['tau'] > 2 ) :
            X2=np.inf
        if (pdict['b'] > 1) or (pdict['b'] < 0):
            X2=np.inf
        if abs(pdict['p'])<0.:
            X2=np.inf
        return X2

    def pL2pdict(self,pL):
        """
        Covert a list of floating parameters to the standard parameter
        dictionary.
        """
        pdict = {}
        i = 0 
        for k in self.pdict.keys():
            if self.fixdict[k]:
                pdict[k]=self.pdict[k]
            else:
                pdict[k]=pL[i]
                i+=1

        return pdict

    def pdict2pL(self,pdict):
        """
        Create a list of the floating parameters
        """
        pL = []
        for k in self.pdict.keys():
            if self.fixdict[k] is False:
                pL += [ pdict[k] ]

        return pL
    
    def MA(self,pdict,t):
        """
        Mandel Agol Model.

        Four free parameters taken from current pdict


        pL can either have
           3 parameters : p,tau,b
           2 parameters : p,tau (b is taken from self.fixb)
        """
        pMA3 = [ pdict[k] for k in 'p,tau,b'.split(',') ] 
        res = keptoy.MA(pMA3, self.climb, t - pdict['dt'], usamp=5)
        return res

def TM_read_h5(h5):
    """
    Read in the information from h5 dataset and return TransitModel
    Instance.
    """    
    attrs = dict(h5.attrs)

    if attrs['P'] < 50:
        bPF  = h5['blc10PF0'][:]
        t    = bPF['tb']
        f    = bPF['med']
        ferr = bPF['std'] / np.sqrt( bPF['count'] )
        b1   = bPF['count']==1 # for 1 value std ==0 which is bad
        ferr[b1] = ma.median(ma.masked_invalid(bPF['std']))
    else:
        lc   = h5['lcPF0'][:]
        t    = lc['tPF']
        f    = lc['f']
        ferr = np.std( (f[1:]-f[:-1])*np.sqrt(2))
        ferr = np.ones(lc.size)*ferr
    try:
        p0 = np.sqrt(1e-6*attrs['df'])
    except:
        p0 = np.sqrt(attrs['mean'])

    # Initial guess for MA parameters.
    pdict=dict(p=p0,tau= attrs['tdur']/2.,b=.3,dt=0.) 
    # Find global best fit value
    trans = TransitModel(t,f,ferr,attrs['climb'],pdict)
    return trans



def TM_to_h5(trans,h5):
    """
    Save results from the fit into an h5 group.
    """
    fitgrp = h5.create_group('fit')
    fitgrp['fit'] = trans.MA(trans.pdict,trans.t)

    fitgrp.create_group('pdict')
    dict2group(h5,'fit/pdict',trans.pdict)

    dsL = 't,f,ferr,uncert,chain,fits'.split(',')
    for ds in dsL:
        if hasattr(trans,ds):
            fitgrp[ds]  = getattr(trans,ds)

def TM_getMCMCdict(h5):
    """
    Returns a dictionary with the best fit MCMC parameters.
    """
    keys = 'p,tau,b'.split(',')
    ucrt = pd.DataFrame(h5['fit/uncert'][:],index=['15','50','85'])[keys].T
    ucrt['med'] = ucrt['50']
    ucrt['sig'] = (ucrt['85']-ucrt['15'])/2
    
    if ucrt.ix['b','85']-ucrt.ix['b','50'] > ucrt.ix['b','15']:
        ucrt.ix['b','sig'] =None
        ucrt.ix['b','med'] =ucrt.ix['b','85']

    d = {}
    for k in keys:
        d[k]     = ucrt.ix[k,'med']
        d['u'+k] = ucrt.ix[k,'sig']

    for k in 'skic,P,t0'.split(','):   
        d[k] = h5.attrs[k]
    return d

def TM_unitsMCMCdict(d0):
    """
    tau e[day] -> tau[hrs]
    p [frac] -> p [percent]
    """
    d = copy.copy(d0)
    
    d['p']    *= 100.
    d['up']   *= 100.
    d['tau']  *= 24.
    d['utau'] *= 24.
    return d 

def TM_stringMCMCdict(d0):
    """
    Convert MCMC parameter output to string
    """
    d = copy.copy(d0) 
    keys = 'p,tau,b'.split(',')    
    for k in keys:
        for k1 in [k,'u'+k]:
            d[k1] = "%.2f" % d0[k1]

    if np.isnan(d0['ub']):
        d['b']  = "<%(b)s" % d
        d['ub'] = ""

    return d
