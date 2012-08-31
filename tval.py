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
from scipy import optimize
from scipy import ndimage as nd
import qalg

import copy
import keptoy
import tfind
from numpy.polynomial import Legendre
from matplotlib import mlab
import h5py
import os
from matplotlib.cbook import is_string_like,is_numlike


import config

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
    this is.  For now, I've just compensated by tPF+=keptoy.lc


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

    Returns
    -------
    tLbl     : index of the candence closest to the transit.
    tRegLbl  : numerical labels for transit region starting at 0
    cRegLbl  : numerical labels for continuum region starting at 0

    """

    t = t.copy()
    t += t0shft(t,P,t0)

    names = ['tRegLbl','cRegLbl','tLbl']
    rec  = np.zeros(t.size,dtype=zip(names,[int]*3) )
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
        
        rec['tRegLbl'][bt] = iTrans
        rec['cRegLbl'][bc] = iTrans
        rec['tLbl'][it]    = iTrans

        iTrans += 1 
        tmdTrans = iTrans * P

    return rec

def LDT(t,fm,cLbl,tLbl):
    """
    Local Detrending

    A simple function that subtracts a polynomial trend from the
    continuum region on either side of a transit.  There must be at
    least two valid data points on either side of the transit to
    include it in the fit.

    Parameters
    ----------
    t    : time
    fm   : masked flux array
    cLbl : Labels for continuum regions.  Computed using `transLabel`
    cLbl : Labels for transit regions.  Computed using `transLabel`
    
    Returns
    -------
    fldt : local detrended flux
    """

    fldt    = ma.masked_array( np.zeros(t.size) , True ) 
    assert type(fm) is np.ma.core.MaskedArray,'Must have mask'
    
    for i in range(cLbl.max() + 1):
        # Points corresponding to continuum region.
        bc = (cLbl == i ) 
        fc = fm[bc]       
        tc = t[bc]

        # Identifying the detrending region.
        bldt = (t > tc[0]) & (t < tc[-1])
        tldt = t[bldt]

        # Times of transit
        bt = (tLbl == i)  
        tt = t[bt]

        # There must be a critical number of valid points before and
        # after the transit.
        ncBefore = fc[ tc < tt[0] ].count()  
        ncAfter  = fc[ tc < tt[-1] ].count()  

        if (ncBefore > 2) & (ncAfter > 2) :        
            shft = - np.mean(tc)
            tc   -= shft 
            tldt -= shft
            pfit = np.polyfit(tc,fc,1)
            fldt[bldt] = fm[bldt] - np.polyval(pfit,tldt)
            fldt.mask[bldt] = fm.mask[bldt]
        else:
            print "trans # %i failed: ncB %i ncA %i" % (i,ncBefore,ncAfter)

    return fldt

def PF(t,fm,P,t0,tdur,cfrac=3,cpad=1):
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
    tudr : transit width (days)

    Returns
    -------
    tPF  : Phase folded time.  0 corresponds to mid transit.
    fPF  : Phase folded flux.
    """

    rec = transLabel(t,P,t0,tdur,cfrac=cfrac,cpad=cpad)
    tLbl,cLbl = rec['tRegLbl'],rec['cRegLbl']

    fldt     = LDT(t,fm,cLbl,tLbl)

    tm = ma.masked_array(t,fldt.mask,copy=True)
    dt = t0shft(t,P,t0)
    tm += dt

    tPF = np.mod(tm[~tm.mask]+P/2,P)-P/2
    fPF = fldt[~fldt.mask] 
    sid = np.argsort(tPF)

    tPF = tPF[sid]
    fPF = fPF[sid]
    
    bg  = ~np.isnan(tPF) & ~np.isnan(fPF)
    tPF = tPF[bg]
    fPF = fPF[bg]
    return tPF,fPF

class Peak():
    noPrintRE = '.*?file|climb|skic'
    noDBRE    = 'climb'
    def __init__(self,*args,**kwargs):
        """
        Peak Object

        Can intialize from mqcal,grid,db files or from a previous pk file.
        """
        if len(args) is 3:
            attrs = {}
            attrs['mqcalfile'] = args[0]
            attrs['gridfile']  = args[1]
            attrs['dbfile']    = args[2]

        self.ds = {}    
        if len(args) is 1:
            self.pkfile    = args[0]
            hpk            = h5py.File(self.pkfile)
            attrs          = dict(hpk.attrs) 
            self.ds = {}
            for k in hpk.keys():
                self.ds[k] = hpk[k][:]
            hpk.close()
        # If the quick kwarg is set, return the attributes specified in h5file.
        try:
            if kwargs['quick']:
                self.attrs = attrs
                return
        except KeyError:
            pass

        hlc = h5py.File(attrs['mqcalfile'],'r+')
        hgd = h5py.File(attrs['gridfile'],'r+')
        self.lc  = hlc['LIGHTCURVE'][:]
        self.res = hgd['RES'][:]
        hlc.close()
        hgd.close()

        # Add important values to attrs dict
        if attrs.keys().count('sKIC') == 0:
            skic = os.path.basename(attrs['gridfile']).split('.')[0]
            attrs['skic'] = skic
        if attrs.keys().count('climb') == 0:
            # Pull limb darkening coeff from database
            con = sqlite3.connect(attrs['dbfile'])
            cur = con.cursor()
            cmd = "select a1,a2,a3,a4 from b10k where skic='%s' " % attrs['skic']
            cur.execute(cmd)
            attrs['climb'] = np.array(cur.fetchall()[0])
        
        if len(args) is 3:
            # Pull the peak information
            rgpk = gridPk(self.res)
            rpk = rgpk[-2:-1]
            for k in ['t0','Pcad','twd','mean','s2n','noise']:
                attrs[k] = rpk[k][0]
            attrs['tdur'] = attrs['twd']*keptoy.lc
            attrs['P']    = attrs['Pcad']*keptoy.lc
            attrs['skic'] = skic

        # Add commonly used values as attrs
        self.t    = self.lc['t']
        self.fm   = ma.masked_array(self.lc['fcal'],self.lc['fmask'])
        self.P    = attrs['P']
        self.t0   = attrs['t0']
        self.tdur = attrs['tdur']
        self.attrs = attrs

    def phaseFold(self):
        """
        Phase Fold

        Add the phase-folded light curve to the peak object.
        """
        attrs = self.attrs
        for ph in [0,180]:
            P,tdur = attrs['P'],attrs['tdur']
            t0     = attrs['t0'] + ph / 360. * P 
            tPF,fPF = PF(self.t,self.fm,P,t0,tdur)
            tPF += keptoy.lc
            lcPF = np.array(zip(tPF,fPF),dtype=[('tPF',float),('fPF',float)] )
            self.ds['lcPF%i' % ph] = lcPF

    def fit(self):
        """
        Fit MA model to PF light curves
        
        Noise attribute is equivalent to a noise per transit.  
        noise per point is ~ np.sqrt(twd) * noise
        """
        attrs = self.attrs
        pL0 = [np.sqrt(attrs['mean']),attrs['tdur']/2.,.3 ]
        for ph in [0,180]:
            PF = self.ds['lcPF%i' % ph]
            def model(pL):
                return keptoy.MAfast(pL,attrs['climb'],PF['tPF'],usamp=11)
            def obj(pL):
                res = (PF['fPF']-model(pL))/(attrs['noise']*attrs['twd'])
                return (res**2).sum()/PF.size
            pL1,fopt,iter,warnflag,funcalls  \
                = optimize.fmin(obj,pL0,full_output=True) 
            fit = model(pL1)
            self.ds['lcPF%i' % ph] = mlab.rec_append_fields(PF,'fit',fit)
            self.attrs['pL%i' % ph]  = pL1
            self.attrs['X2_%i' % ph] = fopt
            
    def s2ncut(self):
        """
        Cut out the transit and recomput S/N.  It s2ncut should be low.
        """

        lbl = transLabel(self.t,self.P,self.t0,self.tdur)
        attrs = self.attrs
        # Notch out the transit and recompute
        fmcut = self.fm.copy()
        fmcut.mask = fmcut.mask | (lbl['tRegLbl'] >= 0)
        dMCut = tfind.mtd(self.t,fmcut, attrs['twd'] )    

        Pcad0 = np.floor(attrs['Pcad'])
        r = tfind.ep(dMCut, Pcad0)

        i = np.nanargmax(r['mean'])
        if i is np.nan:
            s2n = np.nan
        else:
            s2n = r['mean'][i]/attrs['noise']*np.sqrt(r['count'][i])

        self.attrs['s2ncut'] = s2n
        
    def write(self,pkfile,**kwargs):
        hpk = h5plus.File(pkfile,**kwargs)

        for k in self.attrs.keys():
            hpk.attrs[k] = self.attrs[k]
        for k in self.ds.keys():
            hpk.create_dataset(k,data=self.ds[k])
        hpk.close()
    
    def flatten(self,exclRE):
        """
        Return a flat dictionary with exclRE keys excluded.
        """
        pkeys = [k for k in self.attrs.keys() if re.match(exclRE,k) is None]
        d = {}
        for k in pkeys:
            v = self.attrs[k]
            if k.find('pL') != -1:
                suffix = k.split('pL')[-1]
                d['df'+suffix]  = v[0]
                d['tau'+suffix] = v[1]
                d['b'+suffix]   = v[2]
            else:
                d[k] = v
        return d
                        
    def pL2d(self,pL):
        return dict(df=pL[0],tau=pL[1],b=pL[2])

        
    def __str__(self):
        dprint = self.flatten(self.noPrintRE)
        strout = \
"""
%s
-------------
""" % self.attrs['skic']
        for k in dprint.keys():
            v = dprint[k]
            strout += '%s %.4g\n' % (k.ljust(8),v)

        return strout

    def get_db(self):
        """
        Return a dict to store as db
        """
        d = self.flatten(self.noDBRE)

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



def window(fl,PcadG):
    """
    Compute the window function.

    The fraction of epochs that pass our criteria for transit.
    """

    winL = []
    for Pcad in PcadG:
        flW = tfind.XWrap(fl,Pcad,fill_value=False)
        win = (flW.sum(axis=0) >= 3).astype(float)
        npass = np.where(win)[0].size
        win =  float(npass) / win.size
        winL.append(win)

    return winL

def harmSearch(res,t,fm,Pcad0,ver=False):
    """
    Harmonic Search

    Look for harmonics of P by computing MES at 1/4, 1/3, 1/2, 2, 3, 4.

    Parameters
    ----------
    
    res   : result from the grid searh
    t     : time
    fm    : flux
    Pcad0 : initial period in
    
    Returns
    -------

    rLarr : Same format as grid output.  If the code conveged on the
            correct harmonic, the true peak will be in here.
    """

    harmL = np.hstack([1,config.harmL])

    imax = 4    
    i = 0 # iteration counter
    bfund = False # Is Pcad0 the fundemental?

    while (bfund is False) and (i < imax):
        def harmres(h):
            Ph = Pcad0 * h
            pkwd = Ph / t.size # proxy for peak width
            pkwdThresh = 0.03
            if pkwd > pkwdThresh:
                dP = np.ceil(pkwd / pkwdThresh)
            else: 
                dP = 1

            P1 = np.floor(Ph) - dP
            P2 = np.ceil(Ph)  + dP

            isStep = np.zeros(fm.size).astype(bool)
            twdG = [3,5,7,10,14,18]
            rtd = tfind.tdpep(t,fm,P1,P2,twdG)
            r   = tfind.tdmarg(rtd)
            return r

        rL = map(harmres,harmL)
        hs2n = [np.nanmax(r['s2n']) for r in rL]
        rLarr = np.hstack(rL)
        
        hs2n = ma.masked_invalid(hs2n)
        hs2n.fill_value=0
        hs2n = hs2n.filled()

        if ver is True:
            np.set_printoptions(precision=1)
            print harmL*Pcad0*keptoy.lc
            print hs2n

        if np.argmax(hs2n) == 0:
            bfund =True
        else:
            jmax = np.nanargmax(rLarr['s2n'])
            Pcad0 = rLarr['Pcad'][jmax]

        i+=1

    return rLarr



def tdict(d,prefix=''):
    """
    
    """
    outcol = ['P','epoch','df','tdur']

    incol = [prefix+oc for oc in outcol]

    outd = {}
    for o,c in zip(outcol,incol):
        try:
            outd[o] = d[c]
        except KeyError:
            pass

    return outd

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

