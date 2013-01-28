"""
Transit Validation

After the brute force period search yeilds candidate periods,
functions in this module will check for transit-like signature.
"""
from scipy.spatial import cKDTree
import sqlite3
import h5plus
import re

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
import emcee

def gridPk(rg,width=1000):
    """
    Grid Peak
    
    Find the peaks in the MES periodogram.

    Parameters
    ----------
    rg    : record array corresponding to the grid output.

    width : Width of the maximum filter used to compute the peaks.
            The peaks must be the highest point in width Pcad.  Pcad
            is not a uniformly sampled grid so actuall peak width (in
            units of days) will vary.

    Returns
    -------
    rgpk : A trimmed copy of rg corresponding to peaks.  We sort on
           the `s2n` key

    TODO
    ----
    Make the peak width constant in terms of days.

    """
    
    s2nmax = nd.maximum_filter(rg['s2n'],width)

    # np.unique returns the (sorted) unique values of the maximum filter.  
    # rid give the indecies of us2nmax to reconstruct s2nmax
    # np.allclose(us2nmax[rid],s2nmax) evaluates to True

    us2nmax,rid = np.unique(s2nmax,return_inverse=True)    
    bins = np.arange(0,rid.max()+2)
    count,bins = np.histogram(rid,bins )
    pks2n = us2nmax[ bins[count>=width] ]
    pks2n = np.rec.fromarrays([pks2n],names='s2n')
    rgpk  = mlab.rec_join('s2n',rg,pks2n) # The join cmd orders by s2n
    return rgpk

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

def pkInfo(lc,res,rpk,climb):
    """
    Peak Information

    Parameters
    ----------
    lc  : light curve record array
    res : grid search array
    rpk : dictionary. peak info.  P, t0, tdur, (all in days)

    Returns
    -------
    LDT      : Locally detrended light curve at transit.
    MA       : Mandel Agol Coeff
    MA_X2    : Mandel Agol Coeff
    
    LDT180   : Locally detrened light curve 180 deg out of phase
    MA180    : Mandel Agol Coeff
    MA180_X2 : Mandel Agol Coeff

    madSES   : MAD of SES over entire light curve
    maSES   : value of MAD SES for the least noisy quarter


    Notes 
    -----
    phase folded time seemed to be too low by 1 cadence.  Track why
    this is.  For now, I've just compensated by tPF+=config.lc
    """

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
#   import pdb;pdb.set_trace()
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

def findPeak(h5):
    # Pull the peak information
    res = h5.RES
    id  = np.argmax(res['s2n'])        
    for k in ['t0','Pcad','twd','mean','s2n','noise']:
        h5.attrs[k] =  res[k][id]
    h5.attrs['tdur'] = h5.attrs['twd']*config.lc
    h5.attrs['P']    = h5.attrs['Pcad']*config.lc

def conv(h5):
    """
    Store variables accessed frequently at the top level.
    """

    h5.tdurcad  = int (h5.attrs['tdur'] / config.lc)
    lc = h5.lc
    h5.fm =  ma.masked_array(lc['fcal'],lc['fmask'])
    h5.dM = tfind.mtd(lc['t'],h5.fm,h5.tdurcad)
    h5.t  = lc['t']

def checkHarmh5(h5):
    """
    If one of the harmonics has higher MES, update P,epoch
    """
    P = h5.attrs['P']

    harmL,MESL = checkHarm(h5.dM,P,ver=False)
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

    lc = h5['mqcal'][:]
    t  = lc['t']
    fm = ma.masked_array(lc['f'],lc['fmask'])

    keys  = ['LDT_deg','cfrac','cpad','nCont']
    kw    = {}
    for k in keys:
        kw[k] = attrs[k]

    P,tdur = attrs['P'],attrs['tdur']
    t0     = attrs['t0'] + ph / 360. * P 
    h5['lcPF%i' % ph] = PF(t,fm,P,t0,tdur,**kw)

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
    
def at_fit(h5,bPF,fitgrp,runmcmc=False,fixb=None):
    """
    Fit MA model binned phase folded light curves.

    xdata : tb  - time at the center of the bin
    ydata : med - binned median
    err   : std / sqrt(count). Note this should probably be higher for
            median, but it just introduces a scalar
            
    Parameters
    ----------

    h5     : h5 file handel
    bPF    : binned lightcurve dataset 
    fitgrp : empty group to store fitted parameters
    """

    attrs    = dict(h5.attrs)
    ph       = bPF.attrs['ph']
    bPFattrs = dict(bPF.attrs) 

    try:
        p0 = np.sqrt(1e-6*attrs['df'])
    except:
        p0 = np.sqrt(attrs['mean'])

    pL0 = [ p0, attrs['tdur']/2. ,.3  ]



    t   = bPF['tb']
    y   = bPF['med']
    err = bPF['std'] / np.sqrt( bPF['count'] )

    b1  = bPF['count'] ==1 # for 1 value std ==0 which is bad
    err[b1] = ma.median(ma.masked_invalid(bPF['std']))

    # Find global best fit value
    trans = TransitModel(t,y,err,attrs['climb'],pL0,fixb=fixb,dt=0)
    trans.register()
    pL1 = trans.fit()
    np.set_printoptions(precision=3)
    fitgrp['fit'] = trans.MA(pL1,trans.t)
    fitgrp.attrs['pL%i'  % ph] = pL1
    fitgrp.attrs['X2_%i' % ph] = trans.chi2(pL1)

    print pL1

    if runmcmc:        
        print "running MCMC"

        # Burn in
        nwalkers = 10; ndims = 3
        p0      = np.vstack([pL1]*nwalkers)
        p0[:,0] += 1e-4*rand.randn(nwalkers)
        p0[:,1] += 1e-2*pL1[1]*rand.random(nwalkers)
        p0[:,2] = 0.8*rand.random(nwalkers) + .1

        sampler = emcee.EnsembleSampler(nwalkers,ndims,trans)
        nburn = 100
        pos, prob, state = sampler.run_mcmc(p0, nburn)

        # Real run
        sampler.reset()
        niter=500
        foo = sampler.run_mcmc(pos, niter, rstate0=state)
        
        uncert = np.percentile(sampler.flatchain,[15,50,85],axis=0)
        fitgrp.attrs['upL%i' % ph]  = uncert
        fitgrp['chain'] = sampler.flatchain 

        nsamp = 200
        ntrial = sampler.flatchain.shape[0]
        id = np.random.random_integers(0,ntrial-1,nsamp)
        yL = [ trans.MA(sampler.flatchain[i],trans.t) for i in id ] 
        yL = np.vstack(yL) 
        fitgrp['fits'] = yL



def at_med_filt(h5):
    """Add median detrended lc"""
    lc = h5['mqcal']
    fm = ma.masked_array(lc['fcal'],lc['fmask'])

    t  = lc['t']
    P  = h5.attrs['P']
    t0 = h5.attrs['t0']
    tdur= h5.attrs['tdur']

    y = h5.fm-nd.median_filter(h5.fm,size=h5.tdurcad*3)
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

    h5.attrs['s2ncut'] = s2n


def at_SES(h5):
    """
    Look at individual transits SES.

    Using the global overall period as a starting point, find the
    cadence corresponding to the peak of the SES timeseries in
    that region.
    """
    t = h5.t
    attrs = h5.attrs
    rLbl = transLabel(t,attrs['P'],attrs['t0'],attrs['tdur']*2)
    qrec = keplerio.qStartStop()
    q = np.zeros(t.size) - 1
    for r in qrec:
        b = (t > r['tstart']) & (t < r['tstop'])
        q[b] = r['q']

    tRegLbl = rLbl['tRegLbl']
    dM = h5.dM
    # Find the index of the SES peak.
    btmid = np.zeros(tRegLbl.size,dtype=bool) # True if transit mid point.
    for iTrans in range(0,tRegLbl.max()+1):
        tRegSES = dM[ tRegLbl==iTrans ] 
        if tRegSES.count() > 0:
            maSES = np.nanmax( tRegSES.compressed() )
            btmid[(dM==maSES) & (tRegLbl==iTrans)] = True

    tnum = tRegLbl[btmid]
    ses  = dM[btmid]
    q    = q[btmid]
    season = np.mod(q,4)

    dtype = [('ses',float),('tnum',int),('season',int)]
    rses = np.array(zip(ses,tnum,season),dtype=dtype )

    h5['SES'] = rses
    h5['tRegLbl'] = tRegLbl
    ses_o,ses_e = ses[season % 2  == 1],ses[season % 2  == 0]
    h5.attrs['SES_even'] = np.median(ses_e) 
    h5.attrs['SES_odd']  = np.median(ses_o) 
    h5.attrs['KS_eo']    = ks_2samp(ses_e,ses_o)[1]

    for i in range(4):
        ses_i = ses[season % 4  == i]
        ses_not_i = ses[season % 4  != i]
        h5.attrs['SES_%i' %i] = np.median(ses_i)
        h5.attrs['KS_%i' %i]  = ks_2samp(ses_i,ses_not_i)[1]

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

def __str__(h5):
    dprint = h5.flatten(h5.noPrintRE)
    return h5.dict2str(dprint)

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
        if v<1e-3:
            vstr = '%.4g [ppm] \n' % (v*1e6)
        else: 
            vstr = '%.4g \n' % v
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
    dP    = np.abs(phase-d['phase']) 
    dP    = np.vstack([dP,1-dP])
    dP    = dP.min(axis=0)      

    iphase_close = np.argmin(dP)
    phase_close  = phase[iphase_close]

    # s2n for the closest twd,P,phas
    noise = ma.median( ma.abs(dM) )   
    s2nF = meanF / noise *np.sqrt(countF) 
    s2nP = s2nF[iPcad_close] # length Pcad0 array with s2n for all the
                             # P at P_closest
    s2n_close =  s2nP[iphase_close]    

    return phase,s2nP,twd_close,P_close,phase_close,s2n_close


def checkHarm(dM,P,harmL=config.harmL,ver=True):
    """
    Evaluate S/N for (sub)/harmonics
    """
    MESL  = []
    Pcad  = P / config.lc

    dMW = FFA.XWrap(dM , Pcad)
    for harm in harmL:
        # Fold dM on the harmonic
        dMW_harm = FFA.XWrap(dM,Pcad*float(harm) )
        sig  = dMW_harm.mean(axis=0)
        c    = dMW_harm.count(axis=0)
        MES = sig * np.sqrt(c) 
        MESL.append(MES.max())

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
    Simple class for fitting transit    
    """
    def __init__(self,t,y,err,climb,pL0,fixb=None,dt=0):
        """
        t : time array
        y : flux array
        err : errors
        climb : size 4 array with non-linear limb-darkening components
        pL0 :  [ Rp/Rstar, tau, b]. If fixb is true, the value of b is ignored
        """
        self.t     = t
        self.y     = y
        self.err   = err
        self.climb = climb
        self.pL0   = pL0

        b         = np.vstack( map(np.isnan, [t,y,err]) ) 
        b         = b.astype(int).sum(axis=0) == 0 
        self.tm   = t[b]
        self.ym   = y[b]
        self.errm = err[b]

        self.dt   = dt
        self.fixb = fixb

    def register(self):
        """
        Starting with pL0, find the optimum displacement
        """
        tau0  = self.pL0[1]
        dt_per_lc = 3

        dtarr = np.linspace(-2*tau0,2*tau0,4*tau0/keptoy.lc*dt_per_lc)
        def f(dt):
            self.dt = dt
            res = optimize.fmin(self.chi2,self.pL0,disp=False,full_output=True) 
            return res[1]

        X2L = np.array( map(f,dtarr) )
        self.dt = dtarr[np.argmin(X2L)]
        print "dt is now",self.dt

    def fit(self):
        if self.fixb != None:
            pL0 = self.pL0[:2]
        else:
            pL0 = self.pL0

        pL1 = optimize.fmin(self.chi2,pL0,disp=False)

        if self.fixb != None:
            pL1 = np.hstack( [ pL1[:2], self.fixb ] ) 

        return pL1

    def __call__(self,pL):
        """Used for emcee MCMC routine"""
        return -self.chi2(pL)

    def chi2(self,pL):
        y_model = self.MA(pL,self.tm)
        resid   = (self.ym - y_model)/self.errm
        X2 = (resid**2).sum()
        return X2
    
    def MA(self,pL,t):
        """
        Mandel Agol Model.

        pL can either have
           3 parameters : p,tau,b
           2 parameters : p,tau (b is taken from self.fixb)
        """
        if self.fixb != None:
            pL = np.hstack( [pL, self.fixb] )
            
        res = keptoy.MA(pL, self.climb, t-self.dt, usamp=5)
        if pL[2] > 1:
            res+=1
        return res



# Commonly used functions for dealing with cuts
### Mstar, 
G = 6.672e-8 # [cm3 g^-1 s^-2]
Rsun = 6.955e10 # cm
Rearth = 6.3781e8 # cm
Msun = 1.9891e33 # [g]
AU   = 1.49597871e13 # [cm]
sec_in_day = 86400


def trimDV(df):
    """
    
    """
    
def addCuts(df):
    """
    Add the following DV statistics (ratios of other DV statistics)
    """
    Rstar = df.Rstar * Rsun # Rstar [cm]
    Mstar = df.Mstar * Msun # Mstar [g]
    P = df.P_out*sec_in_day # P [s]

    a = (P**2*G*Mstar / 4/ np.pi**2)**(1./3) # in cm 
    df['a/R*'] = a/Rstar
    P = df.P_out*sec_in_day # P [s]
    tauMa =  Rstar*P/2/np.pi/a
    df['tauMa'] = tauMa / sec_in_day # Maximum tau assuming circular orbit
    df['taur']  = df.tau0 / df.tauMa

    df['s2n_out_on_in'] = df.s2ncut / df.s2n
    df['med_on_mean']   = df.medSNR / df.s2n
    df['Rp']            = np.sqrt(df.df0)*df.Rstar*109.04
    df['phase_out']     = np.mod(df.t0/df.P_out,1)
#    df['phase_inp']     = df['phase']
    
    return df

def applyCuts(df,cuts):
    """
    Apply cuts

    One by one, test if star passed a particular cut
    bDV is true if star passed all cuts
    """

    cutkeys = []
    for name in cuts['name']:
        cut = cuts.ix[np.where(cuts.name==name)[0]]
        hi = cut['upper']
        lo = cut['lower']
        if np.isnan(hi):
            hi=np.inf
        if np.isnan(lo):
            lo=-np.inf

        cutk = 'b'+name
        cutkeys.append(cutk)
        df[cutk]=(df[name] > lo) & (df[name] < hi) 

    df['bDV'] = df[cutkeys].T.sum()==len(cutkeys)
    return df

def found(df):
    """
    Did we find the transit
    
    Test that the period and phase peak is the same as the input
    
    Parameters
    ----------
    df  : DataFrame with the following columns defined
          - P_inp
          - P_out
          - phase_inp
          - phase_out

    Returns
    -------
    found column
    """

    dP     = np.abs( df.P_inp     - df.P_out     )
    dphase = np.abs( df.phase_inp - df.phase_out )
    dphase = np.min( np.vstack([ dphase, 1-dphase ]),axis=0 )
    dt0    = dphase*df.P_out
    df['found'] = (dP <.1) & (dt0 < .1) 
    return df
