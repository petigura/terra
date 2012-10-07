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

import FFA
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

class Peak(h5plus.File):
    noPrintRE = '.*?file|climb|skic'
    noDiagRE  = '.*?file|climb|skic|KS.|Pcad|X2.|mean_cut|.*?180'
    noDBRE    = 'climb'

    def __init__(self,*args,**kwargs):
        """
        Peak Object

        Can intialize from mqcal,grid,db files or from a previous pk file.
        """        
        if len(args) is 1:
            h5plus.File.__init__(self,args[0],overwrite=False)
        elif len(args) is 4:
            h5plus.File.__init__(self,args[0],overwrite=False)
            mqcalfile,gridfile,dbfile = args[1],args[2],args[3]
            self.attrs['mqcalfile'] = mqcalfile
            self.attrs['gridfile']  = gridfile
            self.attrs['dbfile']    = dbfile

            self['lc'] = h5py.File(mqcalfile)['LIGHTCURVE'][:]
            self['res'] =  h5py.File(gridfile)['RES'][:]

            # Add star name
            bname = os.path.basename(gridfile).split('.')[0]
            if len(bname) == 9:
                skic = bname
            elif len(bname) == 16:
                skic = bname[:9]

            self.attrs['skic'] = skic
            
            # Add limb darkening coeffs
            con = sqlite3.connect(dbfile)
            cur = con.cursor()
            cmd = "select a1,a2,a3,a4 from b10k where skic='%s' " % skic
            cur.execute(cmd)
            self.attrs['climb'] = np.array(cur.fetchall()[0])


            # Pull the peak information
            rgpk = gridPk(self['res'][:])
            rpk = rgpk[-1:]
            for k in ['t0','Pcad','twd','mean','s2n','noise']:
                self.attrs[k] = rpk[k][0]
            self.attrs['tdur'] = self.attrs['twd']*config.lc
            self.attrs['P']    = self.attrs['Pcad']*config.lc

        try:
            kwargs['quick']
        except KeyError:
            # Add commonly used values as attrs
            lc = self['lc'][:]
            self.res   = self['res'][:]
            self.t     = lc['t']
            self.fm    = ma.masked_array(lc['fcal'],lc['fmask'])
            self.P     = self.attrs['P']
            self.t0    = self.attrs['t0']
            self.df    = self.attrs['mean']
            self.tdur  = self.attrs['tdur']
            self.tdurcad = int(np.round(self.tdur / config.lc))
            self.dM   = tfind.mtd(self.t,self.fm,self.tdurcad)



    def checkHarm(self):
        """
        If one of the harmonics has higher MES, update P,epoch
        """
        harmL,MESL = checkHarm(self.dM,self.P,ver=False)
        idma = np.nanargmax(MESL)
        if idma != 0:
            harm = harmL[idma]
            print "%s P has higher MES" % harm
            self.attrs['P']    *= harm
            self.attrs['Pcad'] *= harm
            self.P    *= harm

    def at_phaseFold(self):
        """ Add locally detrended light curve"""
        attrs = self.attrs
        for ph in [0,180]:
            P,tdur = attrs['P'],attrs['tdur']
            t0     = attrs['t0'] + ph / 360. * P 
            tPF,fPF = PF(self.t,self.fm,P,t0,tdur)
            tPF += keptoy.lc
            lcPF = np.array(zip(tPF,fPF),dtype=[('tPF',float),('fPF',float)] )
            self['lcPF%i' % ph] = lcPF
            lcPF = self['lcPF%i' % ph]
            x,y = lcPF['tPF'],lcPF['fPF']

            for nbpt in [1,5]: # Number of Bins Per Transit
                bins = self.get_bins(x,nbpt)
                y = ma.masked_invalid(y)
                bx,by = hbinavg(x[~y.mask],y[~y.mask],bins)
                self['bx%i_%i'% (ph,nbpt)] = bx
                self['by%i_%i'% (ph,nbpt)] = by

    def at_med_filt(self):
        """Add median detrended lc"""
        y = self.fm-nd.median_filter(self.fm,size=self.tdurcad*3)
        self['fmed'] = y

        # Shift t-series so first transit is at t = 0 
        dt = t0shft(self.t,self.P,self.t0)
        tf = self.t + dt
        phase = np.mod(tf+self.P/4,self.P)/self.P-1./4
        x = phase * self.P
        
        self['tPF'] = x

        # bin up the points
        for nbpt in [1,5]:
            bins = self.get_bins(x,nbpt)
            y = ma.masked_invalid(y)
            bx,by = hbinavg(x[~y.mask],y[~y.mask],bins)
            self['bx%i'%nbpt] = bx
            self['by%i'%nbpt] = by

            bout = np.abs(bx) > self.tdur
            noise = np.std(by[bout])
            self.attrs['std_out%i' % nbpt] = noise
            self.attrs['max_out%i' % nbpt] = np.max(by[bout])
            self.attrs['min_out%i' % nbpt] = np.min(by[bout])
            for per in [5,50,95]:
                k = 'p%i_out%i'% (per,nbpt)
                self.attrs[k] = np.percentile(by[bout],per)

            self.attrs['s2n%i' % nbpt] = self.df / noise


    def get_bins(self,x,nbpt):
        """Return bins of a so that nbpt fit in a transit"""
        return np.linspace(x.min(),x.max(),
                           np.round(x.ptp()/self.tdur*nbpt) +1 )

    def at_fit(self):
        """
        Fit MA model to PF light curves
        
        Noise attribute is equivalent to a noise per transit.  
        noise per point is ~ np.sqrt(twd) * noise
        """
        attrs = self.attrs

        tdur = self.attrs['mean']
        mean = self.attrs['tdur']
        pL0 = [np.sqrt(attrs['mean']),attrs['tdur']/2.,.3,0]
        for ph in [0,180]:
            PF = self['lcPF%i' % ph][:]
            def model(pL):
                return keptoy.MAfast(pL[:3],attrs['climb'],PF['tPF']-pL[-1],usamp=11)
            def obj(pL):
                res = (PF['fPF']-model(pL))/(attrs['noise']*attrs['twd'])
                return (res**2).sum()/PF.size
            pL1,fopt,iter,warnflag,funcalls  \
                = optimize.fmin(obj,pL0,full_output=True,disp=False) 
            fit = model(pL1)
            self['lcPF%i' % ph] = mlab.rec_append_fields(PF,'fit',fit)
            self.attrs['pL%i' % ph]  = pL1
            self.attrs['X2_%i' % ph] = fopt
            
    def at_s2ncut(self):
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


    def at_SES(self):
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

        self['SES'] = rses
        ses_o,ses_e = ses[season % 2  == 1],ses[season % 2  == 0]
        self.attrs['SES_even'] = np.median(ses_e) 
        self.attrs['SES_odd']  = np.median(ses_o) 
        self.attrs['KS_eo']    = ks_2samp(ses_e,ses_o)[1]

        for i in range(4):
            ses_i = ses[season % 4  == i]
            ses_not_i = ses[season % 4  != i]
            self.attrs['SES_%i' %i] = np.median(ses_i)
            self.attrs['KS_%i' %i]  = ks_2samp(ses_i,ses_not_i)[1]

    def at_rSNR(self):
        """
        Robust Signal to Noise Ratio
        """
        ses = self['SES'][:]['ses'].copy()
        ses.sort()
        self.attrs['clipSNR'] = np.mean(ses[:-3]) / self.attrs['noise'] *np.sqrt(ses.size)
        x = np.median(ses) 
        self.attrs['medSNR'] =  np.median(ses) / self.attrs['noise'] *np.sqrt(ses.size)

    def at_all(self):
        self.checkHarm()
        self.at_phaseFold()
        self.at_fit()
        self.at_med_filt()
        self.at_s2ncut()
        self.at_SES()
        self.at_rSNR(self)
            
    def at_s2n_known(self,d):
        """
        When running a simulation, we know a priori where the transit
        will fall.  This function attaches the s2n_known given the
        closest P,t0,twd
        """                
        tup = s2n_known(d,self.t,self.fm)
            
        self.attrs['twd_close']   = tup[2]
        self.attrs['P_close']     = tup[3]
        self.attrs['phase_close'] = tup[4]
        self.attrs['s2n_close']   = tup[5]
        
        

    def plot_diag(self):
        """
        Print a 1-page diagnostic plot of a given pk.

        """
        fig = plt.figure(figsize=(20,12))
        gs = GridSpec(8,10)
        axGrid     = fig.add_subplot(gs[0,0:8])
        axStack    = fig.add_subplot(gs[3: ,0:8])
        axPFAll    = fig.add_subplot(gs[1,0:8])
        axPF       = fig.add_subplot(gs[2,0:4])
        axPF180    = fig.add_subplot(gs[2,4:8],sharex=axPF,sharey=axPF)
        axScar     = fig.add_subplot(gs[0,-1])
        axSES      = fig.add_subplot(gs[1,-1])
        axSeason   = fig.add_subplot(gs[2,-1])
        axAutoCorr = fig.add_subplot(gs[3,-1])

        plt.sca(axGrid)
        self.plotGrid()
        at = AnchoredText('Periodogram',prop=tprop, frameon=True,loc=2)
        axGrid.add_artist(at)
        axGrid.xaxis.set_ticks_position('top')
        plt.title('Period (days)')
        plt.ylabel('MES')

        plt.sca(axPFAll)
        plt.plot(self['tPF'],self['fmed'],',',alpha=.5)
        plt.plot(self['bx1'],self['by1'],'o',mew=0)
        plt.plot(self['bx5'],self['by5'],'.',mew=0)
        y = self['fmed']
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
        rses = self['SES']
        axSES.plot(rses['tnum'],rses['ses']*1e6,'.')
        axSES.xaxis.set_visible(False)
        at = AnchoredText('Transit SES',prop=tprop, frameon=True,loc=2)
        axSES.add_artist(at)

        axSeason.plot(rses['season'],rses['ses']*1e6,'.')
        axSeason.xaxis.set_visible(False)
        at = AnchoredText('Season SES',prop=tprop, frameon=True,loc=2)
        axSeason.add_artist(at)

        plt.sca(axAutoCorr)
        self.plotAutoCorr()
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        
        plt.gcf().text( 0.75, 0.05, self.diag_leg() , size=10, name='monospace',
                        bbox=dict(visible=True,fc='white'))
        plt.tight_layout()
        plt.gcf().subplots_adjust(hspace=0.01,wspace=0.01)

    ###########################
    # Helper fuctions to plot #
    ###########################
    def plotAutoCorr(self):
        bx5fft = np.fft.fft( self['by5'][:].flatten() )
        corr = np.fft.ifft( bx5fft*bx5fft.conj()  )
        plt.plot(np.roll(corr,corr.size/2))
        plt.xlabel('Displacement')

    def plotPF(self,ph):
        PF      = self['lcPF%i' % ph]
        x,y,yfit = PF['tPF'],PF['fPF'],PF['fit']
        plt.plot(x,y,',',color='k')
        plt.plot(x,yfit,lw=3,color='c')        

        x,y = self['bx%i_%i'% (ph,5)],self['by%i_%i'% (ph,5)]
        plt.plot(x,y,'o',mew=0,color='red')



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
        for k in self.keys():
            hpk.create_dataset(k,data=self[k])
        hpk.close()
    
    def flatten(self,exclRE):
        """
        Return a flat dictionary with exclRE keys excluded.
        """
        pkeys = self.attrs.keys() 
        pkeys = [k for k in pkeys if re.match(exclRE,k) is None]
        pkeys.sort()

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
            if v<1e-3:
                vstr = '%.4g [ppm] \n' % (v*1e6)
            else: 
                vstr = '%.4g \n' % v
            strout += k.ljust(12) + vstr

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
    Pcad  = P / keptoy.lc

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

