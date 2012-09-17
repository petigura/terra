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
from scipy.stats import ks_2samp

import qalg

import copy
import keptoy
import tfind
from numpy.polynomial import Legendre
from matplotlib import mlab
import h5py
import os
import matplotlib
envlist = os.environ.keys()
if envlist.count('PBS_QUEUE')!=0:
    matplotlib.use('Agg')
from matplotlib.cbook import is_string_like,is_numlike
from matplotlib.pylab import plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import sketch
tprop = dict(size=10,name='monospace')
import keplerio

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

    Notes
    -----
    tLbl might not be the best way to find the mid transit index.  In
    many cases, dM[rec['tLbl']] will decrease with time, meaning there
    is a cumulative error that's building up.
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

def LDT(t,fm,cLbl,tLbl,verbose=False):
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

    verbose
         : list bad transit.

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
        elif verbose==True:
            print "no data for trans # %i" % (i)
        else:
            pass
        
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
    noDiagRE  = '.*?file|climb|skic|KS.|Pcad|X2.|mean_cut|.*?180'
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
            rpk = rgpk[-1:]
            for k in ['t0','Pcad','twd','mean','s2n','noise']:
                attrs[k] = rpk[k][0]
            attrs['tdur'] = attrs['twd']*keptoy.lc
            attrs['P']    = attrs['Pcad']*keptoy.lc
            attrs['skic'] = skic

        # Add commonly used values as attrs
        self.t     = self.lc['t']
        self.fm    = ma.masked_array(self.lc['fcal'],self.lc['fmask'])
        self.P     = attrs['P']
        self.t0    = attrs['t0']
        self.tdur  = attrs['tdur']
        self.attrs = attrs
        self.tdurcad = int(np.round(self.tdur / config.lc))
        self.dM   = tfind.mtd(self.t,self.fm,self.tdurcad)

    def phaseFold(self):
        """
        Phase Fold

        Locally detrend each transit.  Store a piece of transit for
        quick look.
        """
        attrs = self.attrs
        for ph in [0,180]:
            P,tdur = attrs['P'],attrs['tdur']
            t0     = attrs['t0'] + ph / 360. * P 
            tPF,fPF = PF(self.t,self.fm,P,t0,tdur)
            tPF += keptoy.lc
            lcPF = np.array(zip(tPF,fPF),dtype=[('tPF',float),('fPF',float)] )
            self.ds['lcPF%i' % ph] = lcPF

    def binPhaseFold(self):
        for ph in [0,180]:
            lcPF = self.ds['lcPF%i' % ph]
            x,y = lcPF['tPF'],lcPF['fPF']
            bv = ~np.isnan(x) & ~np.isnan(y)

            # Remove the nans from phase folded light curve.
            try:
                assert x[bv].size==x.size
            except AssertionError:
                print 'nans in the arrays.  Removing them.'
                x = x[bv]
                y = y[bv]
                yfit = yfit[bv]

            def bavg(nb_per_trans):
                bins = np.linspace(x.min(),x.max(),
                                   np.round(x.ptp()/self.tdur*nb_per_trans) +1 )
                bx,by = hbinavg(x,y,bins)
                dtype = [('tPF',float),('fPF',float)]
                r = np.array(zip(bx,by),dtype=dtype)
                return r

            self.ds['lgbinPF%i' %ph ] = bavg(1)        
            self.ds['smbinPF%i' %ph ] = bavg(5)

    def med_filt(self):
        from scipy.ndimage import median_filter
        y = self.fm-median_filter(self.fm,size=self.tdurcad*3)
        self.ds['fmed'] = y

        # Shift t-series so first transit is at t = 0 
        dt = t0shft(self.t,self.P,self.t0)
        tf = self.t + dt
        phase = np.mod(tf+self.P/4,self.P)/self.P-1./4
        x = phase * self.P
        
        self.ds['tPF'] = x

        # bin up the points
        for nb_per_trans in [1,5]:
            bins = np.linspace(x.min(),x.max(),
                               np.round(x.ptp()/self.tdur*nb_per_trans) +1 )

            y = ma.masked_invalid(y)
            bx,by = hbinavg(x[~y.mask],y[~y.mask],bins)
            self.ds['bx%i'%nb_per_trans] = bx
            self.ds['by%i'%nb_per_trans] = by


            bout = np.abs(bx) > self.tdur
            self.attrs['std_out%i' % nb_per_trans] = np.std(by[bout])
            self.attrs['max_out%i' % nb_per_trans] = np.max(by[bout])
            self.attrs['min_out%i' % nb_per_trans] = np.min(by[bout])
            self.attrs['p5_out%i'  % nb_per_trans]  = np.percentile(by[bout],5)
            self.attrs['p5_out%i'  % nb_per_trans]  = np.percentile(by[bout],5)



    def cut_stat(self):
        lgbinPF = self.ds['lgbinPF0']
        lgbx,lgby = lgbinPF['tPF'],lgbinPF['fPF']
        lgmbx = ma.masked_inside(lgbx,-2*self.tdur,2*self.tdur)
        lgmby = ma.masked_array(lgby,lgmbx.mask)
        self.attrs['mean_cut'] = ma.mean(lgmby)
        self.attrs['std_cut']  = ma.std(lgmby)

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
                = optimize.fmin(obj,pL0,full_output=True,disp=False) 
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


    def SES(self):
        """
        Look at individual transits SES.

        Using the global overall period as a starting point, find the
        cadence corresponding to the peak of the SES timeseries in
        that region.
        """
        t = self.t
        rLbl = transLabel(t,self.P,self.t0,self.tdur*2)
        qrec = keplerio.qStartStop()
        q = np.zeros(t.size) - 1
        for r in qrec:
            b = (t > r['tstart']) & (t < r['tstop'])
            q[b] = r['q']

        tRegLbl = rLbl['tRegLbl']
        dM = self.dM
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

        self.ds['SES'] = rses
        ses_o,ses_e = ses[season % 2  == 1],ses[season % 2  == 0]
        self.attrs['SES_even'] = np.median(ses_e) 
        self.attrs['SES_odd']  = np.median(ses_o) 
        self.attrs['KS_eo']    = ks_2samp(ses_e,ses_o)[1]

        for i in range(4):
            ses_i = ses[season % 4  == i]
            ses_not_i = ses[season % 4  != i]
            self.attrs['SES_%i' %i] = np.median(ses_i)
            self.attrs['KS_%i' %i]  = ks_2samp(ses_i,ses_not_i)[1]
        
    def plot_diag(self):
        fig = plt.figure(figsize=(20,12))
        gs = GridSpec(8,10)
        axGrid  = fig.add_subplot(gs[0,0:8])
        axStack = fig.add_subplot(gs[3: ,0:8])
        axPFAll = fig.add_subplot(gs[1,0:8])
        axPF    = fig.add_subplot(gs[2,0:4])
        axPF180 = fig.add_subplot(gs[2,4:8],sharex=axPF,sharey=axPF)
        axScar  = fig.add_subplot(gs[0,-1])
        axSES   = fig.add_subplot(gs[1,-1])
        axSeason= fig.add_subplot(gs[2,-1])
        axAutoCorr = fig.add_subplot(gs[3,-1])

        plt.sca(axGrid)
        self.plotGrid()
        at = AnchoredText('Periodogram',prop=tprop, frameon=True,loc=2)
        axGrid.add_artist(at)
        axGrid.xaxis.set_ticks_position('top')
        plt.title('Period (days)')
        plt.ylabel('MES')

        plt.sca(axPFAll)
        plt.plot(self.ds['tPF'],self.ds['fmed'],',',alpha=.5)
        plt.plot(self.ds['bx1'],self.ds['by1'],'o',mew=0)
        plt.plot(self.ds['bx5'],self.ds['by5'],'.',mew=0)
        y = self.ds['fmed']
        axPFAll.set_ylim( (np.percentile(y,5),np.percentile(y,95) ) )


        plt.sca(axPF180)
        self.plotPF(180)
        cax = plt.gca()
        cax.xaxis.set_visible(False)
        cax.yaxis.set_visible(False)
        at = AnchoredText('Phase Folded LC + 180',prop=tprop,frameon=True,loc=2)
        cax.add_artist(at)

        plt.sca(axPF)
        self.plotPF(0)
        cax = plt.gca()
        cax.xaxis.set_visible(False)
        cax.yaxis.set_visible(False)
        at = AnchoredText('Phase Folded LC',prop=tprop,frameon=True,loc=2)
        cax.add_artist(at)
        df = self.attrs['pL0'][0]**2
        plt.ylim(-5*df,3*df)

        plt.sca(axStack)
        self.plotSES()
        plt.xlabel('Phase')
        plt.ylabel('SES (ppm)')

        plt.sca(axScar)
        sketch.scar(self.res)
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)

        axSeason.set_xlim(-1,4)
        rses = self.ds['SES']
        axSES.plot(rses['tnum'],rses['ses']*1e6,'.')
        axSES.xaxis.set_visible(False)
        at = AnchoredText('Transit SES',prop=tprop, frameon=True,loc=2)
        axSES.add_artist(at)

        axSeason.plot(rses['season'],rses['ses']*1e6,'.')
        axSeason.xaxis.set_visible(False)
        at = AnchoredText('Season SES',prop=tprop, frameon=True,loc=2)
        axSeason.add_artist(at)

        bx5fft = np.fft.fft( self.ds['by5'].flatten() )
        corr = np.fft.ifft( bx5fft*bx5fft.conj()  )
        axAutoCorr.plot(np.roll(corr,corr.size/2))

        


        plt.gcf().text( 0.75, 0.05, self.diag_leg() , size=12, name='monospace',
                    bbox=dict(visible=True,fc='white'))

        plt.tight_layout()
        plt.gcf().subplots_adjust(hspace=0.01,wspace=0.01)

    ###########################
    # Helper fuctions to plot #
    ###########################
    def plotPF(self,ph):
        PF      = self.ds['lcPF%i' % ph]
        smbinPF = self.ds['lgbinPF%i' % ph]

        # Plot phase folded LC
        x,y,yfit = PF['tPF'],PF['fPF'],PF['fit']
        plt.plot(x,y,',',alpha=.5)
        plt.plot(x,yfit,alpha=.5)        
        plt.axhline(0,alpha=.3)
        plt.plot(smbinPF['tPF'],smbinPF['fPF'],'o',mew=0,color='red')

    def plotSES(self):
        df = self.attrs['pL0'][0]**2
        sketch.stack(self.t,self.dM*1e6,self.P,self.t0,step=3*df*1e6)
        plt.autoscale(tight=True)

    def plotGrid(self):
        x = self.res['Pcad']*config.lc
        plt.plot(x,self.res['s2n'])
        id = np.argsort( np.abs(x - self.P) )[0]
        plt.plot(x[id],self.res['s2n'][id],'ro')
        plt.autoscale(tight=True)

    ######
    # IO #
    ######

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
                d['df'+suffix]  = v[0]**2
                d['tau'+suffix] = v[1]
                d['b'+suffix]   = v[2]
            else:
                d[k] = v
        return d
                        
    def pL2d(self,pL):
        return dict(df=pL[0],tau=pL[1],b=pL[2])
        
    def __str__(self):
        dprint = self.flatten(self.noPrintRE)

    def diag_leg(self):
        dprint = self.flatten(self.noDiagRE)
        return self.dict2str(dprint)

    def dict2str(self,dprint):
        """
        """

        strout = \
"""
%s
-------------
""" % self.attrs['skic']

        keys = dprint.keys()
        keys.sort()
        for k in keys:
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


################################################################
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

