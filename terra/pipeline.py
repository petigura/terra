"""Pipeline

Defines the components of the TERRA pipeline.

"""

import copy
import os

import numpy as np
from numpy import ma
import h5py
from utils.h5plus import h5F
import pandas as pd
from matplotlib import pylab as plt
from matplotlib import mlab

from plotting import kplot, tval_plotting
import prepro
import tfind
import tval
import keptoy
import config
from k2utils import photometry
import transit_model as tm
from utils import h5plus
import copy
import tval
from lmfit import minimize, fit_report

deltaPcad = 10
 
from utils.hdfstore import HDFStore

class Pipeline(HDFStore):
    """Initialize a pipeline model.

    The pipeline object itself is just a container object that can easily write
    to hdf5 using pandas. Different codes can perform module operations on the
    pipeline object.
    
    Args:
        lc (Optional[pandas.DataFrame]): Light curve. Must have the 
            following columns: t, f, ferr, fmask. Setting equal to None is 
            done for reading from disk
        header (Optional[dict]): metadata to be stored with the
            pipeline object. At a bare minimum, it must include
            the star name

    Example:
    
        # Working with the pipeline
        >>> pipe = Pipeline(lc=lc, starname='temp',header)
        >>> pipeline.preprocess(pipe) 
        >>> pipeline.grid_search(pipe) 
        >>> pipeline.data_validation(pipe)
        >>> pipeline.fit_transits(pipe)
    """
    lc_required_columns = ['t','f','ferr','fmask']
    pgram_nbins = 2000 # Bin the periodogram down to save storage space
    def __init__(self, lc=None, starname=None, header=None):
        super(Pipeline,self).__init__()
        if type(lc)==type(None):
            return 

        for col in self.lc_required_columns:
            assert list(lc.columns).index(col) >= 0, \
                "light curve lc must contain {}".format(col)

        self.update_header('starname', starname, 'String Star ID')
        for key,value in header.iteritems():
            self.update_header(key, value,'')
        self.update_header('finished_preprocess',False,'preprocess complete?')
        self.update_header('finished_grid_search',False,'grid_serach complete?')
        self.update_header(
            'finished_data_validation',False,'Data validation complete?'
        )
        self.update_table('lc',lc,'light curve')

    def _get_fm(self):
        """Convenience function to return masked flux array"""
        fm = ma.masked_array(
            self.lc.f.copy(), self.lc.fmask.copy(), fill_value=0 )
        fm -= ma.median(fm)
        return fm

def read_hdf(hdffile, group):
    pipe = Pipeline()
    pipe.read_hdf(hdffile, group)
    return pipe

def preprocess(pipe):
    """Process light curve in the time domain

    Args:
        pipe (Pipeline object)

    Returns:
        None 

    """
    fm = pipe._get_fm()
    isOutlier = prepro.isOutlier(fm, [-1e3,10], interp='constant')
    pipe.lc['isOutlier'] = isOutlier
    pipe.lc['fmask'] = fm.mask | isOutlier | np.isnan(fm.data)
    print "preprocess: identified {} outliers in the time domain".format(
          isOutlier.sum() )
    print "preprocess: {} measurements, {} are masked out".format(
        len(pipe.lc) , pipe.lc['fmask'].sum())

    pipe.update_header('finished_preprocess',True)
    return

def grid_search(pipe, P1=0.5, P2=None, periodogram_mode='max'):
    """Run the grid based search

    Args:
        P1 (Optional[float]): Minimum period to search over. Default is 0.5
        P2 (Optional[float]): Maximum period to search over. Default is half 
            the time baseline
        **kwargs : passed to grid.periodogram

    Returns:
        None

    """
    if type(P2) is type(None):
        P2 = 0.49 * pipe.lc.t.ptp() 

    t = np.array(pipe.lc.t)
    fm = pipe._get_fm() 
    grid = tfind.Grid(t, fm)
    pipe.update_header('dt',grid.dt,'Exposure time (days)')
    tbase = pipe.lc.t.max() - pipe.lc.t.min()
    pgram_params = tfind.periodogram_parameters(P1, P2 , tbase, nseg=10)
    pgram = grid.periodogram(pgram_params, mode=periodogram_mode)
    pgram = pgram.query('P > 0') # cut out candences that failed

    if len(pgram) > pipe.pgram_nbins:
        log10P = np.log10(pgram.P)
        bins = np.logspace(log10P.min(),log10P.max(),pipe.pgram_nbins)
        pgram['Pbin'] = pd.cut(
            pgram.P, bins, include_lowest=True, precision=4,labels=False
            )

        # Take the highest s2n row at each period bin
        pgram = pgram.sort_values(['Pbin','s2n']).groupby('Pbin').last()
        pgram = pgram.reset_index()

    row = pgram.sort_values('s2n').iloc[-1]
    pipe.update_header('grid_s2n', row.s2n, "Periodogram peak s2n")
    pipe.update_header('grid_P', row.P, "Periodogram peak period")
    pipe.update_header('grid_t0', row.t0, "Periodogram peak transit time")
    pipe.update_header(
        'grid_tdur', row.tdur, "Periodogram peak transit duration"
    )
    pipe.update_table('pgram',pgram,'periodogram')
    pipe.update_header('finished_grid_search',True)
    print row
    return None

def fit_transits(pipe):
    """Fit transits

    Performs the following tasks:
        1. Performs local detrending
        2. Fits a light curve with constant ephemeris
        3. Fits signle transits starting with 2., allowing transit times to vary
        4. Fits signle transits starting with 2., allowing Rp/Rstar to vary

    Args:
        pipe (Pipeline object): pipeline objec

    Notes:
        Tables updated:
            - lc
        Tables added:
            - lcdt
            - lcfit
            - transits


    Returns:
        None

    """
    batman_kw = dict(supersample_factor=4, exp_time=1/48.)
    label_transit_kw = dict(cpad=0, cfrac=2)
    local_detrending_kw = dict(poly_degree=1, label_transit_kw=label_transit_kw)

    # Compute initial parameters. Fits are more robust if we star with
    # transits that are too wide as opposed to to narrow
    P = pipe.grid_P
    t0 = pipe.grid_t0 
    tdur = pipe.grid_tdur * 2 
    rp = np.sqrt(pipe.pgram.sort_values('s2n').iloc[-1]['mean'])
    b = 0.5

    # Grab data, perform local detrending, and split by tranists.
    lcdt = pipe.lc.copy()
    lcdt = lcdt[~lcdt.fmask].drop(['ftnd_t_roll_2D'],axis=1)
    lcdt = tval.local_detrending(lcdt, P, t0, tdur, **local_detrending_kw)
    lcdt['ferr'] = lcdt.query('continuum==1').f.std()
    time_base = np.mean(lcdt['t'])
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)
    ferr = np.array(lcdt.ferr)

    # Perform global fit. Set some common-sense limits on parameters
    ntransits = P / lcdt.t.ptp()
    tm = tval.TransitModel(P, t0 - time_base, rp, tdur, b, )
    tm.lm_params['rp'].min = 0.0
    tm.lm_params['tdur'].min = 0.0
    tm.lm_params['b'].min = 0.0
    tm.lm_params['b'].max = 1.0
    tm.lm_params['t0'].min = tm.lm_params['t0'] - tdur 
    tm.lm_params['t0'].max = tm.lm_params['t0'] + tdur 
    tm.lm_params['per'].min = tm.lm_params['per'] - tdur / ntransits
    tm.lm_params['per'].max = tm.lm_params['per'] + tdur / ntransits

    tm_initial = copy.deepcopy(tm)
    out = minimize(
        tm.residual, tm.lm_params, args=(t, f, ferr), method='nelder'
    )
    tm_global = copy.deepcopy(tm)
    print "Global fit values"
    print fit_report(out)

    # Store best fit parameters
    par = tm.lm_params
    pipe.update_header('fit_P', par['per'].value, "Best fit period")
    pipe.update_header('fit_uP', par['per'].stderr, "Uncertainty")
    t0 = par['t0'].value + time_base
    pipe.update_header('fit_t0', t0,  "Best fit transit mid-point" )
    pipe.update_header('fit_rp', par['rp'].value, "Best fit Rp/Rstar")
    pipe.update_header('fit_tdur', par['tdur'].value,  
        "Best fit transit duration" 
    )
    pipe.update_header('fit_b', par['b'].value, "Best fit impact parameter")
    for k in 't0 rp tdur b'.split():
        pipe.update_header('fit_u{}'.format(k), par[k].stderr, "Uncertainty")
        
    # Add in best-fit lightcurve
    lcfit = pipe.lc.copy()
    tfit = np.linspace(pipe.lc.t.min(), pipe.lc.t.max(), len(pipe.lc) * 4 )
    lcfit = pd.DataFrame(tfit,columns=['t'])
    lcfit['t_shift'] = lcfit.t - time_base
    lcfit['f'] = tm.model(tm.lm_params, np.array(lcfit.t_shift))
    lcfit['f_initial'] = tm.model(tm_initial.lm_params, np.array(lcfit.t_shift))

    # Fit inidvidual transits
    transits = []
    for transit_id, idx in lcdt.groupby('transit_id').groups.iteritems():
        transit = dict(transit_id=transit_id)
        lcdt_single = lcdt.ix[idx]
        t = np.array(lcdt_single.t_shift) 
        f = np.array(lcdt_single.f)
        ferr = np.array(lcdt_single.ferr)
        def _get_tm():
            tm = copy.deepcopy(tm_global)
            for key in 'per t0 rp tdur b'.split():
                tm.lm_params[key].vary = False
            return tm

        # Fit the transit times, holding other parameters constant
        tm = _get_tm()
        tm.lm_params['t0'].vary = True
        out = minimize(tm.residual, tm.lm_params, args=(t,f,ferr) )
        transit['t0'] = out.params['t0'].value
        transit['ut0'] = out.params['t0'].stderr

        # Fit the transit depths, holding other parameters constant
        tm = _get_tm()
        tm.lm_params['rp'].vary = True
        out = minimize(tm.residual, tm.lm_params, args=(t,f,ferr) )
        transit['rp'] = out.params['rp'].value
        transit['urp'] = out.params['rp'].stderr
        transits+=[transit]

    # Constant ephemeris, then adjust for observed O - C.
    fit_P = pipe.fit_P
    fit_t0 = pipe.fit_t0
    transits = pd.DataFrame(transits)
    transits['omc'] = transits['t0'] - tm_global.lm_params['t0']
    transits['t0'] = fit_t0 + transits.transit_id * fit_P + transits['omc']

    def add_phasefold(lc):
        return tval.add_phasefold(lc, lc.t, fit_P, fit_t0)

    lcfit = add_phasefold(lcfit)
    lcdt = add_phasefold(lcdt)
    lc = add_phasefold(pipe.lc)

    pipe.update_table('lc', lc)
    pipe.update_table('lcdt', lcdt, "Same as lc with f detrended")
    pipe.update_table('lcfit', lcfit, "Supersampled best fit light curve.")
    pipe.update_table(
        'transits', transits,
        'Inidividual t0 and rp, holding other parameters fixed.'
    )
    return None

def secondary_eclipse_search(pipe):
    """Search for secondary eclipse
    """
    phase_limit = 0.1
    fm = pipe._get_fm()
    t = np.array(pipe.lc.t)
    transit_mask = np.abs(pipe.lc.phase) < phase_limit
    fm.mask = fm.mask | transit_mask
    grid = tfind.Grid(t, fm)
    Pcad1 = pipe.grid_P / pipe.dt - 1
    Pcad2 = pipe.grid_P / pipe.dt + 1
    twd = pipe.grid_tdur / pipe.dt
    pgram_params = [dict(Pcad1=Pcad1, Pcad2=Pcad2, twdG=[twd])]
    pgram = grid.periodogram(pgram_params,mode='max')
    row = pgram.sort_values('s2n').iloc[-1]

    pipe.update_header('se_s2n',row['s2n'],'Secondary eclipse candidate SNR') 
    pipe.update_header('se_t0',row['t0'],'Secondary eclipse candidate epoch')
    se_phase = (pipe.se_t0 - pipe.grid_t0) / pipe.grid_P 
    pipe.update_header(
        'se_phase',se_phase,'phase offset of secondary eclipse (deg)'
    )

    return None

def bin_phasefold(pipe):
    P = pipe.fit_P
    t0 = pipe.fit_t0
    dt = pipe.dt
    n_phase_bins = np.round(P / dt)
    t_phasefold_bins = np.linspace(-0.5 * pipe.fit_P, 0.5 * P, n_phase_bins)
    lc = pipe.lc.copy()
    lc = lc[~lc.fmask]
    lcpfbin = tval.bin_phasefold(lc, t_phasefold_bins)
    lcdtpfbin = tval.bin_phasefold(pipe.lcdt, t_phasefold_bins)

    pipe.update_table(
        'lcpfbin', lcpfbin, 'phase folded and binned version of lc'
    )
    pipe.update_table(
        'lcdtpfbin', lcdtpfbin, 'phase folded and binned verion of lcdt'
    )
    return None

def autocorr(pipe, clip_factor=3):
    """Compute the auto-correlation of light curve.

    Fold at the best-fitting period and compute the auto
    auto-correlation.

    Args: 
        pipe (pipeline object) : Pipeline objec
        clip_width (float) : 
    """

    # Prepare DataFrames
    lc = pipe.lcpfbin.copy() 
    lc['f'] = lc.fmed.fillna(value=0)
    lcshift = lc.copy()
    auto = pd.DataFrame(np.array(lc.index),columns=['i_shift'])

    t = np.array(lc.t_phasefold)
    dt = np.median(t[1:] - t[:-1]) # Compute spacing between bins

    # Compute auto correlation
    auto['autocorr'] = 0
    for i_shift in auto.i_shift:
        lcshift.index = np.roll(lc.index,i_shift)
        auto.ix[i_shift,'autocorr'] = np.sum(lc.f * lcshift.f)

    i_shift, _, _ = tval.phasefold(auto.i_shift, len(auto.i_shift), 0)
    auto['i_shift'] = i_shift
    auto['t_shift'] = auto.i_shift * dt
    auto = auto.sort_values('i_shift')
    
    idx = auto.autocorr.idxmax()
    clip_width = clip_factor * pipe.header.value.fit_tdur
    idxclip = auto[np.abs(auto['t_shift']) > clip_width].autocorr.idxmax()
    autor = auto.ix[idxclip,'autocorr'] / auto.ix[idx,'autocorr'] 

    pipe.update_table('auto',auto, 'Auto correlation of binned light curve')
    pipe.update_header('autor', autor, 
        'max autocorr divided by max out of transit autocorr'
    )
    return 

def multiCopyCut(file0,file1,pdict=None):
    """
    Multi Planet Copy Cut

    Copys the calibrated light curve to file1. Cuts the transit out.
    """

    with h5py.File(file0) as h5, h5py.File(file1) as h5new:
        h5new.create_group('pp')
        h5new.copy(h5['/pp/mqcal'],'/pp/mqcal')
        lc   = h5['/pp/mqcal'][:]

        if pdict is None:
            print "Cutting out transit (using lcPF0)"
            lcPF = h5['lcPF0'][:]
            j    = mlab.rec_join('t',lc,lcPF,jointype='leftouter')
            addmask = j['tPF']!=0
        else:
            print "Cutting out transit %s " % pdict
            tlbl = tval.transLabel(lc['t'],pdict['P'],pdict['t0'],pdict['tdur'])
            addmask = tlbl['tRegLbl'] >=0

        lc['fmask'] = lc['fmask'] | addmask
        h5new['/pp/mqcal'][:] = lc
