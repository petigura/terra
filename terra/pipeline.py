"""
TERRA
The top level controller.  

To Do:

Make terra object oriented.

lc = 
header = dict(starname='', other meta-data)

pipe.process() # Conditions data in the time domain
pipe.grid_search() # Perform the grid search.
pipe.data_validation()

pipe.to_hdf(outfile) # Should be able to read different checkpoints

terra(t, f, ferr, fmask, plimb=[passed to batman], P1=[passed to grid], P2)
terra(t, f, ferr, fmask, plimb=[passed to batman], P2=)
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
    lc_required_columns = ['t','f','ferr','fmask']
    pgram_nbins = 2000 # Bin the periodogram down to save storage space

    def __init__(self, lc=None, header=dict(starname='starname') ):
        """Initialize a pipeline model.

        Args:
            lc (Optional[pandas.DataFrame]): Light curve. Must have the 
                following columns: t, f, ferr, fmask. Setting equal to None is 
                done for reading from disk
            header (Optional[dict]): metadata to be stored with the
                pipeline object. At a bare minimum, it must include
                the star name
        """

        super(Pipeline,self).__init__()

        if type(lc)==type(None):
            return None 

        for col in self.lc_required_columns:
            assert list(lc.columns).index(col) >= 0, \
                "light curve lc must contain {}".format(col)

        self.update_header(
            'finished_preprocess', False, 
            'Have we preprocessed data in time domain?'
            )
        self.update_header(
            'finished_grid_search', False, 
            'Have we run the grid search?'
            )
        self.update_header(
            'finished_data_validation', False, 
            'Have we completed data validation?'
           )

        self.update_table('lc',lc,'light curve')

    def _get_fm(self):
        """Convenience function to return masked flux array"""
        fm = np.array(self.lc.f)
        fm = ma.masked_array(fm, self.lc.fmask, fill_value=0 )
        fm -= ma.median(fm)
        return fm

    def preprocess(self):
        """Process light curve in the time domain

        Args:
            None

        """
        fm = self._get_fm()
        isOutlier = prepro.isOutlier(fm, [-1e3,10], interp='constant')
        self.lc['isOutlier'] = isOutlier
        self.lc['fmask'] = fm.mask | isOutlier | np.isnan(fm.data)
        print "preprocess: identified {} outliers in the time domain".format(
              isOutlier.sum() )
        print "preprocess: {} measurements, {} are masked out".format(
            len(self.lc) , self.lc['fmask'].sum())

        self.update_header('finished_preprocess',True)

    def grid_search(self, P1=0.5, P2=None, periodogram_mode='max'):
        """Run the grid based search

        Args:
            P1 (Optional[float]) : Minimum period to search over
            P2 
        """

        t = np.array(self.lc.t)
        fm = self._get_fm() 
        grid = tfind.Grid(t, fm)

        tbase = self.lc.t.max() - self.lc.t.min()
        pgram_params = tfind.periodogram_parameters(P1, P2 , tbase, nseg=10)
        pgram = grid.periodogram(pgram_params, mode=periodogram_mode)
        pgram = pgram.query('P > 0') # cut out candences that failed

        if len(pgram) > self.pgram_nbins:
            log10P = np.log10(pgram.P)
            bins = np.logspace(log10P.min(),log10P.max(),self.pgram_nbins)
            pgram['Pbin'] = pd.cut(
                pgram.P, bins, include_lowest=True, precision=4,labels=False
                )
            
            # Take the highest s2n row at each period bin
            pgram = pgram.sort(['Pbin','s2n']).groupby('Pbin').last()
            pgram = pgram.reset_index()

        row = pgram.sort('s2n').iloc[-1]
        self.update_header('grid_P', row.P, "Period with highest s2n")
        self.update_header(
            'grid_t0', row.t0, "Time of transit with highest s2n"
        )
        self.update_header(
            'grid_tdur', row.tdur, "transit duration with highest s2n"
        )
        self.update_table('pgram',pgram,'periodogram')
        self.update_header('finished_grid_search',True)
        print row


#    def data_validation(self, PF_kw=None, 
#                        climb = [0.773, -0.679, 1.14, -0.416] ):
#        
#        if PF_kw==None:
#            PF_kw = {}
#            
#        
#        dv = tval.DV( self.lc.to_records(), self.header['grid_P'], self.pgram.to_records() )
#        dv.climb = np.array( climb )
#        dv.at_phaseFold(0, **PF_kw)
#        dv.at_phaseFold(180, **PF_kw)
#
#        for ph,binsize in zip([0,0,180,180],[10,30,10,30]):
#            dv.at_binPhaseFold(ph,binsize)
#
#        dv.at_s2ncut()
#        dv.at_phaseFold_SecondaryEclipse()
#        dv.at_med_filt()
#        dv.at_autocorr()
#
#        trans = tm.from_dv(dv,bin_period=0.1)
#        trans.register()
#        trans.pdict = trans.fit_lightcurve()[0]
#        dv.trans = trans
#        self.dv = dv
#        self.header['finished_data_validation'] = True
    

def read_hdf(hdffile, group):
    pipe = Pipeline()
    pipe.read_hdf(hdffile, group)
    return pipe


def fit_transits(pipe):
    batman_kw = dict(supersample_factor=4, exp_time=1/48.)
    label_transit_kw = dict(cpad=0, cfrac=2)
    local_detrending_kw = dict(poly_degree=1, label_transit_kw=label_transit_kw)

    # Compute initial parameters. Fits are more robust if we star with
    # transits that are too wide as opposed to to narrow
    P = pipe.grid_P
    t0 = pipe.grid_t0 
    tdur = pipe.grid_tdur * 2 
    rp = np.sqrt(pipe.pgram.sort('s2n').iloc[-1]['mean'])
    b = 0.5


    # Grab data, perform local detrending, and split by tranists.
    lc = pipe.lc.copy()
    lc = lc[~lc.fmask].drop(['ftnd_t_roll_2D'],axis=1)
    lc = tval.local_detrending(lc, P, t0, tdur, **local_detrending_kw)
    lc['ferr'] = lc.query('continuum==1').f.std()
    time_base = np.mean(lc['t'])
    lc['t_shift'] = lc['t'] - time_base
    t = np.array(lc.t_shift)
    f = np.array(lc.f)
    ferr = np.array(lc.ferr)

    # Perform global fit
    tm = tval.TransitModel(P, t0 - time_base, rp, tdur, b, )
    tm.lm_params['rp'].min = 0.0
    tm.lm_params['tdur'].min = 0.0
    tm.lm_params['b'].min = 0.0
    tm.lm_params['b'].max = 1.0
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
    pipe.update_header('fit_b', par['b'].value, "Best fit impact parameter")
    for k in 't0 rp tdur b'.split():
        pipe.update_header('fit_u{}'.format(k), par[k].stderr, "Uncertainty")

    lc0 = pipe.lc.copy()
    lc0['t'] -= time_base
    tfit = np.linspace(lc0.t.min(), lc0.t.max(), len(lc0) * 4 )
    lcfit = pd.DataFrame(tfit,columns=['t'])

    lcfit['f'] = tm.model(tm.lm_params, np.array(lcfit.t))
    lcfit['f_initial'] = tm.model(tm_initial.lm_params, np.array(lcfit.t))
    lcfit['t'] += time_base

    # Add in phase folded time
    fit_P = pipe.fit_P
    fit_t0 = tm.lm_params['t0'].value
    dt = tval.t0shft( np.array(pipe.lc.t), fit_P, fit_t0)
    def t_phasefold(t):
        _t_phasefold = t + dt
        _t_phasefold = np.mod(_t_phasefold + fit_P/2, fit_P) - fit_P/2 
        return _t_phasefold
    lc['t_phasefold'] = t_phasefold(lc.t)
    lcfit['t_phasefold'] = t_phasefold(lcfit.t)
    pipe.update_table('lcdt', lc, "Same as lc with f detrended")
    pipe.update_table('lcfit', lcfit, "Supersampled best fit light curve.")

    # Fit inidvidual transits
    transits = []
    for transit_id, idx in lc.groupby('transit_id').groups.iteritems():
        transit = dict(transit_id=transit_id)
        lc_single = lc.ix[idx]
        t = np.array(lc_single.t_shift) 
        f = np.array(lc_single.f)
        ferr = np.array(lc_single.ferr)
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
    transits = pd.DataFrame(transits)
    transits['omc'] = transits['t0'] - tm_global.lm_params['t0']
    transits['t0'] = fit_t0 + transits.transit_id * fit_P + transits['omc']
    pipe.update_table(
        'transits', transits,
        'Inidividual t0 and rp, holding other parameters fixed.'
    )
    return pipe

        
#def terra():
#    """
#    Top level wrapper for terra, all parameters are passed in as
#    keyword arguments
#
#    """
#    @print_timestamp
#    def pp(self):
#        con = sqlite3.connect(self.pardb)
#        df = pd.read_sql('select * from pp',con,index_col='id')
#        d = dict(df.ix[self.starname])
#        d['outfile'] = self.star_gridfile
#        d['path_phot'] = self.lcfile
#        terra.pp(d)
#
#    @print_timestamp
#    def grid(self,debug=False):
#        con = sqlite3.connect(self.pardb)
#        df = pd.read_sql('select * from grid',con,index_col='id')
#        d = dict(df.ix[self.starname])
#        d['outfile'] = self.star_gridfile
#        if debug:
#            d['p1'] = 1.
#            d['p2'] = 2.
#    
#        terra.grid(d)
#
#    @print_timestamp
#    def data_validation(self):
#        con = sqlite3.connect(self.pardb)
#        df = pd.read_sql('select * from dv',con,index_col='id')
#        d = dict(df.ix[self.starname])
#        d['outfile'] = self.star_gridfile
#        terra.data_validation(d)
#        dscrape = dv_h5_scrape(self.star_gridfile)
#        print pd.Series(dscrape)
#        # insert into sqlite3 database
#        #insert_dict(dscrape, 'candidate', self.tps_resultsdb)
#


def data_validation(par):
    """
    Data Validation
    
    Parameters
    ----------
    par : dictionary with the following keys
          - LDT_deg : polynomial degree of local detrender
          - cfrac   : size of continuum region multiple of tranist durtaion
          - cpad    : size of padding between in/egress and start of
                      continuum region
          - nCont   : minimum number of continuum points to use transit
          - a[1-4]  : limb-darkening coeffs
          - outfile : which h5 file to read
          - skic    : star
          - update  : If True, we can modify the h5 file.

          optional keys
          - P    : Alternative period to fold on
          - t0   : Alt epoch. BJD - 2454833 days. 
                   t=0 <->  12:00 on Jan 1, 2009 UTC
          - tdur : transit durtaion (days)

    Example
    -------

    >>> import terra
    >>> ddv = {'LDT_deg': 3,'cfrac': 3, 'cpad': 0.5, 'nCont': 4, 
              'a1': 0.773,'a2': -0.679,'a3': 1.140, 'a4': -0.416, 
              'outfile':'202073438.grid.h5'}
    >>> terra.data_validation(ddv)

    """


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

####################

def dictSetup(pardict0):
    """
    Check Set necessary fields in pardict
    """
    pardict = copy.copy(pardict0)
    if pardict['rel'] == False:
        pardict = addPaths(pardict)

    # P1,P2 define the range of candences we sample the outlier
    # spectrum at.
    if pardict.has_key('P1') == False:
        pardict['P1'] = int(config.P1 / config.lc)
        pardict['P2'] = int(config.P2 / config.lc)    

    # P1_FFA,P2_FFA define range over which we compute the FFA 
    if ~pardict.has_key('P1_FFA'):
        if pardict['type']=='mcS':
            P1 = int(pardict['inj_P']/config.lc - deltaPcad)
            pardict['P1_FFA'] = max(P1, int(config.P1 / config.lc))

            P2 = int(pardict['inj_P']/config.lc + deltaPcad)
            pardict['P2_FFA'] = min(P2, int(config.P2 / config.lc))
        else:
            pardict['P1_FFA'] = pardict['P1']
            pardict['P2_FFA'] = pardict['P2']
    if pardict['type'] =='mcS':
        assert pardict.has_key('gridfile'),'must point to full res file'
    return pardict

def pardict_print(pardict):
    """Print nicely formatted summary of input/output files"""

    print "Run type:  %(type)s" % pardict
    print "\ninput files:"
    print "-"*50
    print pardict['rawfile']
    print pardict['svd_folder']
    if pardict.has_key('gridfile'):
        print pardict['gridfile']

    print "\noutput files:"
    print "-"*50
    if pardict.has_key('storeGrid'):
        print pardict['storeGrid']
    if pardict.has_key('pngStore'):
        print pardict['pngStore']
    print ""
            
def plot(h5out,pardict):
    import matplotlib
    matplotlib.use('Agg')
    import kplot
    from matplotlib.pylab import plt
    kplot.plot_diag(h5out)
    plt.gcf().savefig(pardict['pngGrid'])

def DVout(h5out,pardict):
    # DV
    DV(h5out,pardict)
    out = tval.flatten(h5out,h5out.noDBRE)
    pL  = h5out['fit'].attrs['upL0'][1]
    out['p0']   = pL[0]
    out['tau0'] = pL[1]
    out['b0']   = pL[2]
    return out

def findItMax(h5):
    # Find maximum iteration 
    itL = [i[0] for i in h5.items() if i[0].find('it')!=-1]
    itMax = np.sort(itL)[-1]        
    return itMax

def inj(h5,pardict):
    """
    Inject Signal


    Parameters
    ----------
    h5       : h5plus object.  Must have /pp/cal dataset
    injdict  : full pardict.  Will strip off only the necessary columns
    """
    injdict = {}
    injkeys = [k for k in pardict.keys() if k.find('inj_') !=-1 ]
    for k in injkeys:
        injdict[k[4:]] = pardict[k]

    for i in range(1,5):
        injdict['a%i' % i]=pardict['a%i' %i]
    
    lc = h5['/pp/cal'][:]
    ft = keptoy.synMA(injdict,lc['t'])
    lc['f'] +=ft
    lc = mlab.rec_append_fields(lc,'finj',ft)

    del h5['/pp/cal']
    h5['/pp/cal'] = lc
    

        
def addPaths(d0):
    """
    Paths are referenced with respect to a base directory, which can
    changed from machine to machine.
    """
    d = copy.copy(d0)
    assert os.environ.has_key('KEPSCRATCH'),'KEPSCRATCH must be defined'
    wkdir = os.environ['KEPSCRATCH']
    for k in ['storeGrid','pngGrid','gridfile','svd_folder','rawfile']:
        if d.has_key(k):
            d[k] = wkdir + d[k]
    return d


def gridShort(h5,pardict):
    """

    """
    with h5py.File(pardict['gridfile'],'r+') as grid0:
    # Only compute the grid over a narrow region in period
        res0 = grid0['it0']['RES'][:]

    h5.attrs['P1_FFA'] = pardict['P1_FFA']
    h5.attrs['P2_FFA'] = pardict['P2_FFA']

    tfind.grid(h5)

    # Add that narrow region to the already computed grid.
    res  = h5['it0']['RES'][:]
    start = np.where(res0['Pcad']==res['Pcad'][0])[0]
    stop  = np.where(res0['Pcad']==res['Pcad'][-1])[0]
    if len(start) > 1:
        start = start[1]
    if len(stop) > 1:
        stop = stop[0]
    res0[start:stop+1] = res
    del h5['it0']['RES']
    h5['it0']['RES'] = res0
    

### Mstar, 
G          = 6.67e-8 # [cm3 g^-1 s^-2]
Rsun       = 6.95e10 # cm
Rearth     = 6.37e8 # cm
Msun       = 1.98e33 # [g]
AU         = 1.50e13 # [cm]
sec_in_day = 86400

def simPts(stars,nsim,limD):
    """
    Simulate Planet Parameters

    Parameters
    ----------
    stars : DataFrame of stellar parameters. Must have the following keys
            - Mstar
            - Rstar
    nsim : Number of simulations
    limD : Specify the range of P and Re

    """
    nsim = int(nsim)
    nstars = len(stars)
    np.random.seed(100)

    # Make a DataFrame with P and Rp
    simPar = {}
    for k in limD.keys():
        lim = limD[k]
        lo,hi = lim[0],lim[1]
        x = np.random.random(nsim)
        x = (np.log10(hi) -np.log10(lo) ) * x + np.log10(lo)
        x = 10**x
        simPar[k] = x
    simPar = pd.DataFrame(simPar)

    idxstars = np.random.random_integers(0,nstars-1,nsim)
    stars = stars.ix[idxstars]
    stars.index = range(nsim)
    simPar = pd.concat([ stars, simPar],axis=1)
    
    # Return a in cm
    def a(x):
        Mstar = Msun*x['Mstar']
        P = x['P']*sec_in_day
        return (G * Mstar * P**2 / (4*np.pi**2))**(1/3.)

    a = simPar.apply(a,axis=1) # [cm]
    n = 2*np.pi / (simPar['P']*sec_in_day) # [s^-1]
    tau = (Rsun*simPar['Rstar'] / a / n)/sec_in_day # [day]
    p = simPar['Re']*Rearth/(simPar['Rstar']*Rsun)

    simPar['inj_tau'] = tau
    simPar['inj_p'] = p
    simPar['id'] = simPar.index
    
    simPar['inj_b'] = np.random.random(nsim)
    simPar['inj_phase'] = np.random.random(nsim) 
    simPar['inj_P'] = simPar['P']

    return simPar

PKG_DIR = os.path.dirname(__file__)
testfitsfn = os.path.join(PKG_DIR,'tests/data/201367065.fits')

def test_terra():    
    dpp = {
        'outfile':'test.h5',
        'path_phot':testfitsfn,
        'type':'tps',
        'update':True,
        'plot_lc':True
        }
    pp(dpp)

    dgrid = {
        'outfile':'test.h5',
        'update':True,
        'P1': 9,
        'P2': 11,
        'fluxField': 'fcal',
        'fluxMask': 'fmask',
        'outfile': 'test.h5',
        'tbase':80
        }

    grid(dgrid)

    ddv = {
        'LDT_deg': 3,
        'cfrac': 3,
        'cpad': 0.5,
        'nCont': 4, 
        'a1': 0.773,
        'a2': -0.679,
        'a3': 1.140,
        'a4': -0.416, 
        'outfile':'test.h5',
        'plot_diag':True,
        }

    data_validation(ddv)

    print "Pipeline ran to competion"
    print "verify that the following files look OK"
    print "\n".join(["test.lc.png","test.pk.png"])




