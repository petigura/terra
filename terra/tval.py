"""
Transit Validation

After the brute force period search yeilds candidate periods,
functions in this module will check for transit-like signature.
"""


import numpy as np
from numpy import ma
from scipy import ndimage as nd
from scipy.spatial import cKDTree
from matplotlib import mlab
import h5py

import pandas as pd

from utils import h5plus
import FFA

import keptoy
import tfind
import keplerio
import config
import batman
from lmfit import Parameters, minimize, fit_report, minimizer
import utils.hdfstore

def phasefold(t, P, t0, starting_phase=-0.5):
    """Return the phase-folded time
    
    Args:
        t (array) : time
        P (float) : Period to fold at
        t0 (float) : reference time. Mapped to phase 0.
        starting_phase (Optional[float]): Lowest phase returned. For example,
            starting_phase = 0.5 runs from -0.5 to 0.5.

    Returns
        t_phasefold (array): Time phase-folded
        phase (array) : phase from 0 to 1.
        cycle (array) : integer labels
    """
    t = np.array(t)
    dt = t0shft( np.array(t), P, t0)
    tshift = t + dt
    t_phasefold = np.mod(tshift - starting_phase*P, P) + starting_phase * P
    phase = t_phasefold / P
    cycle = np.floor(tshift/P - starting_phase).astype(int)
    return t_phasefold, phase, cycle

def add_phasefold(lc, t, P, t0, starting_phase=-0.5):
    """Simple wrapper around phasefold that attaches columns to dataframe

    Args:
        lc : pandas DataFrame
        *args : See `phasefold`
        **kwargs : See `phasefold`

    Returns
        lc (pandas DataFrame) : t_phasefold, phase, cycle keywords are added
    """
    assert isinstance(lc, pd.core.frame.DataFrame),"lc must be DataFrame"
    t_phasefold, phase, cycle = phasefold(
        t, P, t0, starting_phase=starting_phase
    )

    lc['t_phasefold'] = t_phasefold
    lc['phase'] = phase
    lc['cycle'] = cycle
    return lc

def bin_phasefold(lc, t_phasefold_bins):
    """Takes the phase folded light curve and bins up the flux

    Splits the light curve up into specified bins and computes mean,
    median, std, and counts of flux in the given bin.

    Args:
        lc (pandas DataFrame) : Light curve. Must have t_phasefold, and f keys
        t_phasefold_bins : numpy array with the boundaries of the bins

    Returns:
        lcbin (pandas DataFrame) : binned and phase-folded lightcurve

    """

    t_phasefold_bins_center = (
        0.5 * (t_phasefold_bins[:-1] + t_phasefold_bins[1:])
        )

    labels = pd.cut(
        lc.t_phasefold, t_phasefold_bins, labels=t_phasefold_bins_center
    )
    g = lc.groupby(labels, as_index=False )
    lcbin = pd.DataFrame(t_phasefold_bins_center, columns=['t_phasefold'])
    lcbin['fcount'] = g['f'].count()
    lcbin['fmean'] = g['f'].mean()
    lcbin['fmed'] = g['f'].median()
    lcbin['fstd'] = g['f'].std()
    lcbin['ferr'] = lcbin.fstd/np.sqrt(lcbin.fcount)
    return lcbin



class DataValidation(utils.hdfstore.HDFStore):
    """ Object for handling transit validation checks
    
    Args:
        lc (pandas DataFrame) : Light curve. Must have `t`, `f`, and `ferr`, 
            and `fmask`
        P (float) : Period of transit
        t0 (float) : time of transit
        tdur (float) : duration of transit
    """
    def __init__(self, lc, P, t0, tdur, dt):
        # Calls the __init__ constructor of h5plus.iohelper. Then we
        # can do other things
        super(DataValidation,self).__init__()
        self.update_header('P', P, 'Transit period')
        self.update_header('t0', t0, 'Time of transit')
        self.update_header('tdur', tdur, 'Duration of transit')
        tdurcad = int(np.round(tdur / dt))
        self.update_header('tdurcad', tdurcad, 'Transit duration (cadences)')
        self.update_header('dt', dt, 'Observing cadence')
        self.update_table('lc', lc, 'Light curve')

        # Convenience
        self._attach_convenience()
 
    def _attach_convenience(self):
        self.t  = np.array(self.lc['t'])
        self.fm = ma.masked_array(self.lc['f'],self.lc['fmask'])
        self.dM = tfind.mtd(self.fm, self.tdurcad)

    def at_SES(self):
        """
        Attach single event statistics

        Finds the cadence closes to the transit midpoint. For each
        cadence, record in dataset `SES` the following information:

            ses    : Signle event statistic for single transit
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
        t    = self.t
        dM   = self.dM
        rLbl = transLabel(t,self.P,self.t0,self.tdur*2)

        # Hack. K2 doesn't have seasons
        q = np.zeros(t.size) - 1
        season = np.mod(q,4)
        dtype = [('ses',float),('tnum',int),('season',int)]
        dM.fill_value = np.nan
        rses  = np.array(zip(dM.filled(),rLbl['tLbl'],season),dtype=dtype )
        rses  = rses[ rLbl['tLbl'] >= 0 ]

        # If no good transits, break out
        if rses.size==0:
            return

        self.add_dset('rLbl',rLbl,
            description='Transit/continuum labeled (see transLabel doc string')
        self.add_dset('SES',rses,
                      description='Record array with single event statistic')
        self.add_attr('num_trans',rses.size,
                      description='Number of good transits')

        # Median SES, even/odd
        for sfx,i in zip(['even','odd'],[0,1]):
            medses =  np.median( rses['ses'][rses['tnum'] % 2 == i] ) 
            self.add_attr('SES_%s' % sfx, medses,
                          description='Median SES %s' % sfx)

        # Median SES, different seasons
        for i in range(4):
            medses = -99 #Hack
            self.add_attr('SES_%i' % i, medses,
                          description='Median SES [Season %i]' % i )

    def at_phaseFold(self, ph, **PF_kw):
        """ 
        Attach locally detrended light curve

        Parameters
        ----------
        ph : Phase [between 0 and 360]
        PW_kw : dictionary of parameters passed to PF
        """
        # Epoch at arbitrary phase, ph
        t0 = self.t0 + ph / 360. * self.P 
        lcPF = PF(self.t, self.fm, self.P, t0, self.tdur, **PF_kw) 
        lcPF = pd.DataFrame(lcPF)
        self.update_table('lcPF%i' % ph, lcPF,'Phase folded light curve')


    def at_binPhaseFold(self,ph,bwmin):
        """Attach binned phase-folded light curve

        Attaches a dataset named `blc<bwmin>iPF<ph>`  with the following keys:
        - count : Number of each measurment in bin
        - mean : Mean flux value in each bin
        - std : Standard deviation of measurements in each bin
        - med : Median measurement
        - tb : Central time of the bin

        Parameters
        ----------
        ph : Phase [between 0 and 360]
        bwmin - targe bin width (minutes)

        
        Note
        ----
        Must run at_phaseFold first

        dtype=

        """
        # Default dtype
        dtype = [('count', '<f8'), 
                 ('mean', '<f8'), 
                 ('std', '<f8'), 
                 ('med', '<f8'), 
                 ('tb', '<f8')]        

        key = 'lcPF%i' % ph
        d = dict(ph=ph,bwmin=bwmin,key=key)
        desc = 'Binned %(key)s light curve ph=%(ph)i, binsize=%(bwmin)i' % d
        name  = 'blc%(bwmin)iPF%(ph)i' % d

        assert hasattr(self,key),'Must run at_phaseFold first' 
        lcPF = getattr(self,key)
        lcPF = pd.DataFrame(lcPF['tPF f'.split()])
        
        if len(lcPF) < 2:
            print "Phase-folded photometry has less than 2 valid values"
            print "Adding in place holder array and terminating" 
            blcPF = np.zeros(2,dtype)
            self.add_dset(name,blcPF,description=desc)            
            return None

        # Add a tiny bit to xma to get the last element
        bw = bwmin / 60./24. # converting minutes to days
        xmi,xma = lcPF.tPF.min(),lcPF.tPF.max() 
        nbins = int( np.round( (xma-xmi)/bw ) )
        bins = np.linspace(xmi-0.001,xma,nbins+1)
        tb = 0.5*(bins[1:]+bins[:-1])

        # Compute info along columns
        g = lcPF.groupby(pd.cut(lcPF.tPF,bins))
        blcPF = g['f'].agg([np.size, np.mean, np.std, np.median])
        blcPF['tb'] = tb
        blcPF = blcPF.rename(columns={'size':'count','median':'med'})
        blcPF = blcPF.dropna()
        blcPF = blcPF.to_records(index=False)
        self.add_dset(name,blcPF,description=desc)

    def at_s2ncut(self):
        """
        Attach cut s2n statistics
        """

        # Notch out the transit and recompute
        fmcut = self.fm.copy()
        fmcut.fill_value=0
        # Widen by twice the transit duration
        tmask = self.rLbl['tRegLbl'] >= 0
        tmask = np.convolve(
            tmask.astype(float),
            np.ones(self.header['tdurcad'] * 2),
            mode='same'
            )
        tmask = tmask.astype(bool)
        fmcut.mask = fmcut.mask | tmask
        grid = tfind.Grid(self.t,fmcut)


        pgram_params = [
            dict(Pcad1=self.Pcad - 1, Pcad2=self.Pcad + 1, twdG = [self.header['tdurcad']])
        ]
        pgram = grid.periodogram(pgram_params,mode='max')
        idxmax = pgram.s2n.idxmax()

        dkeys = 's2ncut s2ncut_t0 s2ncut_mean'.split()
        pkeys = 's2n t0 mean'.split()

        for dkey,pkey in zip(dkeys,pkeys):
            self.add_attr(dkey,pgram.ix[idxmax,pkey])

    def at_phaseFold_SecondaryEclipse(self):
        """
        Return phasefolded photometry around putative secondary eclipse
        """

        lcPF_SE = PF(self.t,self.fm,self.P,self.s2ncut_t0,self.tdur)
        self.add_dset('lcPF_SE',lcPF_SE,
                      description='Phase folded photometry (secondary eclipse)')
        t0shft_SE = self.t0 - self.s2ncut_t0
        ph = t0shft_SE / self.P * 360
        self.add_attr('t0shft_SE',t0shft_SE,'Time offset of secondary eclipse')
        self.add_attr('ph_SE',ph,'Phase offset of secondary eclipse')

    def at_med_filt(self):
        """Add median detrended lc"""
        fmed = self.fm - nd.median_filter(self.fm, size=self.header['tdurcad']*3)

        # Shift t-series so first transit is at t = 0 
        dt = t0shft(self.t,self.P,self.t0)
        tf = self.t + dt
        phase = np.mod(tf + 0.25 * self.P, self.P) / self.P - 0.25
        tPF = phase * self.P # Phase folded time

        # bin up the points
        for nbpt in [1,5]:
            # Return bins of a so that nbpt fit in a transit
            nbins = np.round( tPF.ptp()/self.tdur*nbpt ) 
            bins =  np.linspace(tPF.min(),tPF.max(),nbins+1)
            fmed = ma.masked_invalid(fmed)
            btPF,bfmed = hbinavg(tPF[~fmed.mask],fmed[~fmed.mask],bins)
            
            rbmed = np.rec.fromarrays([btPF,bfmed],names=['t','f'])

            self.add_dset('rbmed%i' % nbpt, rbmed, description='Binned phase-folded, median filtered timeseries, %i points per tdur'% nbpt) 

        self.add_dset('fmed',fmed,description='Median detrended flux')

    def at_autocorr(self):
        """
        Attach autocorrelation trace
        """
        
        arr = ma.masked_invalid(self.rbmed5['f']  )
        nlag = arr.size
        lag = np.arange(nlag) - nlag/2
        corr = [ma.sum(np.roll(arr,l)*arr) for l in lag]
        corr = np.array(corr)

        b = np.abs(lag > 6) # Bigger than the size of the fit.
        autor = max(corr[~b])/max(np.abs(corr[b]))

        self.add_dset('lag',lag,description='Auto correlation lag')
        self.add_dset('corr',corr,description='Auto correlation amplitude')
        self.add_attr('autor',autor,description=\
            'Ratio of second highest peak to primary autocorrelation peak')


def read_hdf(h5file,group):
    tm = h5plus.read_iohelper(h5file,group)
    tm.__class__ = DV

    # Convenience
    tm._attach_convenience()
    return tm 
    
#
# Helper functions for DV class
#

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

def label_transit(t, P, t0, tdur, cpad=0.5, cfrac=3):
    """Label transits
    
    Given transit ephemerides and time array, assign labels to transit points.

    Args:
        t (numpy.array) : time
        P (float) : Period of transit
        t0 (float) : transit midpoint
        tdur (float) : transit duratoin, i.e. T14
        cfrac (Optional[float]) : length of continuum region, units of tdur.
        cpad (Optional[float]) : padding between nominal end of transit, and 
            beginning of continuum region, units of tdur. 

    Returns:
        labels (record array) : Same length as `t`. Contains the following 
            columns:

            transit_id : integer identification for transit
            transit : 1 if transit, 0 if not
            continuum : 1 if continuum, 0 if not. 
            continuum_before : 1 if continuum before transit, 0 if not.
            continuum_after : 1 if continuum after transit, 0 if not.
    """
    t = t.copy()
    t += t0shft(t,P,t0)

    names = [
        'transit_id','transit','continuum','continuum_before','continuum_after'
    ]

    labels  = np.zeros(t.size, dtype=zip( names, [int] * len(names) ) )
    labels['transit_id'] -= 1
    
    transit_id   = 0 # number of transit, starting at 0.
    tmdTrans = 0 # time of iTrans mid transit time.  
    while tmdTrans < t[-1]:
        # Time since mid transit in units of tdur
        dt = (t - tmdTrans) / tdur
        
        # In transit points
        labels['transit'][ np.abs(dt) < 0.5 ] = 1

        # Continuum points
        continuum_before = ( -0.5 - cpad - cfrac < dt) & (dt < -0.5 - cpad) 
        continuum_after = ( 0.5 + cpad < dt) & (dt < 0.5 + cpad + cfrac) 

        labels['continuum_before'][continuum_before] = 1
        labels['continuum_after'][continuum_after] = 1
        labels['continuum'][continuum_before | continuum_after] = 1
        
        # All transit points
        labels['transit_id'][np.abs(dt) < 0.5 + cpad + cfrac] = transit_id

        transit_id += 1 
        tmdTrans = transit_id * P

    return labels

def local_detrending(lc, P, t0, tdur, poly_degree=3, min_continuum=2, 
                     label_transit_kw=None):
    """
    Local Detrending

    A simple function that subtracts a polynomial trend from the
    continuum region on either side of a transit.  There must be at
    least two valid data points on either side of the transit to
    include it in the fit.

    Args:
        lc (pandas DataFrame) : light curve. Must contain following keys: 
            - t : time 
            - f : flux
            - can contain other keys that come along for the ride.
        P (float) : Period of transit
        t0 (float) : transit midpoint
        tdur (float) : transit duratoin, i.e. T14

        min_continuum (int) : minimum number of continuum points before and 
            after transit to be used. If this is not met, drop the transit

    Returns:
         lcdt : Detrended light curve with the following keys
             - t : time
             - f : detrended flux
             - other keys from transit_label
             - other keys that were along for the rid


    """
    if label_transit_kw==None:
        label_transit_kw = {}

    lcdt = lc.copy()
    t = np.array(lcdt.t)
    labels = label_transit(t, P, t0, tdur, **label_transit_kw)
    labels = pd.DataFrame(labels)

    g = labels.query('transit_id >= 0').groupby('transit_id',as_index=False)
    labels_sum = g.sum()
    transit_id_good = labels_sum[
        (labels_sum.continuum_before > min_continuum) & 
        (labels_sum.continuum_after > min_continuum)
        ].transit_id

    # Reset indecies to join
    labels = labels.reset_index(drop=True)
    lcdt = lcdt.reset_index(drop=True)    
    lcdt = pd.concat([lcdt,labels],axis=1)

    lcdt = lcdt[lcdt.transit_id.isin(transit_id_good)]

    def detrend(x):
        """Function that detrends a single light section of light curve"""

        t = x['t']
        f = x['f']
        t0 = np.mean(x['t'])
        time_since_transit = t - t0

        # select out just the continuum points
        continuum = x['continuum']==1

        pfit = np.polyfit(
            time_since_transit[continuum], f[continuum], poly_degree
            )

        fldt = f.copy()
        fldt -= np.polyval(pfit,time_since_transit)
        return fldt

    g = lcdt.groupby('transit_id')
    fldt = map(detrend,[ lcdt.ix[idx] for idx in g.groups.values() ] )
    fldt = pd.concat(fldt)
    lcdt['f'] = fldt
    return lcdt


class TransitModel(object):
    """Transit Model

    Can compute synthetic transit models and compute residuals to
    observed transit times.

    Args:
        P (float): initial period
        t0 (float): transit center
        rp (float): Rp/Rstar.
        tdur (float): transit duration T14
        b (float): impact parameter
        ecc (float, optional): eccentricity
        w (float, optional): argument of peri (degrees)
        limb_dark (str, optional): type of limb-darkening to use 
            (default: linear)        
        u (list, optional): limb-darkening parameters to use (default: 0.66)
        batman_kw (dict, optional): other keyword arguments to pass to the 
            batman transit model code.
        subtract_one (bool, optional): when computing the model, do we subtract
            unity from light curve.
    """

    def __init__(self, P, t0, rp, tdur, b, ecc=0.0, w=0.0, limb_dark='linear',
                 u=[0.66], subtract_one=True, batman_kw=None):
        if batman_kw==None:
            batman_kw = {}

        lm_params = Parameters()
        lm_params.add('per', value=P, vary=True )
        lm_params.add('t0', value=t0, vary=True )
        lm_params.add('rp', value=rp, vary=True, min=0.0)
        lm_params.add('tdur', value=tdur)
        lm_params.add('b', value=b) 

        self.ecc = ecc
        self.w = w
        self.limb_dark = limb_dark
        self.u = u
        self.lm_params = lm_params
        self.batman_kw = batman_kw
        self.subtract_one = subtract_one

    def model(self, lm_params, t):
        per = lm_params['per'].value
        t0 = lm_params['t0'].value
        rp = lm_params['rp'].value
        tdur = lm_params['tdur'].value
        b = lm_params['b'].value
        bm_params = batman.TransitParams()
        bm_params.per = per
        bm_params.t0 = t0
        bm_params.rp = rp 
        bm_params.a = compute_a(per, tdur, rp, b)
        bm_params.inc = compute_inc(per, tdur, rp, b)
        bm_params.ecc = self.ecc  
        bm_params.w = self.w 
        bm_params.limb_dark = self.limb_dark
        bm_params.u = self.u 
        m = batman.TransitModel(bm_params, t, **self.batman_kw)
        _model = m.light_curve(bm_params) 
        if self.subtract_one:
            _model -= 1
        return _model

    def residual(self, lm_params, t, f, ferr):
        """Normalized residual"""
        _model = self.model(lm_params, t)
        resid = (f - _model) / ferr
        return resid    

def compute_a(P, T14, rp, b):
    """ Compute scaled semi-major axis from P, T14, rp, and b """
    _a = P / np.pi / T14 * np.sqrt( (1+ rp)**2 - b )
    return _a

def compute_inc(P, T14, rp, b):
    """ Compute inclination from P, T14, rp, and b """
    _inc = np.arccos(b / compute_a(P, T14, rp, b) )
    _inc = np.rad2deg(_inc)
    return _inc

def phase_fold(t, fm, P, t0, tdur, cfrac=3, cpad=1, LDT_deg=1, nCont=4):
    """Phase fold light curve
    
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
    deg  : degree of the polynomial fitter (passed to LDT)

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

def findpks(x,y,bins):
    id    = np.digitize(x,bins)
    uid   = np.unique(id)[1:-1] # only elements inside the bin range
    mL    = [np.max(y[id==i]) for i in uid]
    mid   = [ np.where((y==m) & (id==i))[0][0] for i,m in zip(uid,mL) ] 
    return x[mid],y[mid]

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

# 
# Hepler functions no longer used in terra. Not ready to delete yet.
# 

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

