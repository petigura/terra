"""
TERRA
The top level controller.  This function encapsulates the complete pipeline.
"""

import h5py
from h5py import File as h5F 

import numpy as np
from numpy import ma
import glob
import prepro
import tfind
import tval
import keptoy
import config
from matplotlib import mlab
import h5plus
import copy
import os
import pandas as pd
deltaPcad = 10
from config import k2_dir
import photometry
import transit_model as tm

#######################
# Top Level Functions #
#######################

def h5F(par):
    """
    If the update kw is set, open h5 file as a h5plus object.
    """
    outfile = par['outfile']
    if par['update']:
        return h5plus.File(outfile)
    else:
        return h5py.File(outfile)

def pp(par):
    """
    Preprocess

    Parameters
    ----------
    par : dictionary with the following keys
          - rawfile 
          - outfile
          - type = mc/tps
          - update : Overwrite exsiting files? True/False
          - inj_P,inj_phase,inj_p,inj_tau,inj_b      <-- only for mc
          - a1,a2,a3,a4 limb darkening coefficients  <-- inj/rec runs

    Example
    -------

    # Monte Carlo run
    >>> import terra
    >>> dpp = {'a1': 0.77, 'a2': -0.67, 'a3': 1.14,'a4': -0.41,
               'inj_P': 142.035,'inj_b': 0.46,'inj_p': 0.0132,
               'inj_phase': 0.583,'inj_tau': 0.178, 
               'outfile':'temp.grid.h5', 
               'skic': 7831530, 
               'svd_folder':'/global/project/projectdirs/m1669/Kepler/svd/',
               'type': 'mc', 'plot_lc':True}
    >>> terra.pp(dpp)

    # TPS Run (Hacked to work on Ian's photometry)
    dpp = dict(path_phot='photometry/C0_pixdecor/202083828.fits',
             outfile='temp.grid.h5',plot_lc=True,update=True,type='tps')
    terra.pp(dpp)



    Make a dataframe with the following columns
    Minumum processing to look at LC

           skic  
    0   3544595  
    1  10318874  
    2   5735762  
    3  12252424  
    4   4349452  
    5  11295426  
    6  11075737  
    7   2692377  
    8   8753657  
    9  11600889  

    nt['outfile'] = nt.skic.apply(lambda x : "%09d.h5" % x)
    nt['svd_folder']='/global/project/projectdirs/m1669/Kepler/svd/'
    terra.pp(dict(nt.ix[0]))
    """

    par = dict(par) # If passing in pandas series, treat as dict
    print "creating %(outfile)s" % par
    print "Running pp on %s" % par['outfile'].split('/')[-1]

    path_phot = par['path_phot']
    path_phot = os.path.abspath(path_phot) # Expand full path
    
    outfile = par['outfile']
    outfile = os.path.abspath(outfile)

    with h5F(par) as h5:
        lc = photometry.read_photometry(path_phot)

        h5.create_group('pp')
        h5['/pp/cal'] = lc

        if par['type'].find('mc') != -1:
            inj(h5,par)

        # Hack to get around no calibration step
        for k in 'fcal fit fdt'.split():
            lc = mlab.rec_append_fields(lc,k,np.zeros(lc.size))
        lc['fcal'] = lc['f']
        lc['fdt'] = lc['f']

        fcal = ma.masked_array(lc['fcal'],lc['fmask'])
        fcal.fill_value=0

        isOutlier = prepro.isOutlier(fcal.filled(),method='two-sided')
        lc = mlab.rec_append_fields(lc,'isOutlier',isOutlier)
        lc['fmask'] = lc['fmask'] | lc['isOutlier']  | np.isnan(lc['fcal'])

        del h5['/pp/cal'] # Clear group so we can re-write to it.
        h5['/pp/cal'] = lc
        # prepro.cal(h5,par)

        # Store path information
        h5.attrs['phot_basedir'] = os.path.dirname(path_phot)
        h5.attrs['phot_fits_filename'] = os.path.basename(path_phot)

        
        h5.attrs['grid_basedir'] = os.path.dirname(outfile)
        h5.attrs['grid_h5_filename'] = os.path.basename(outfile)

        if par.has_key('plot_lc'):
            if par['plot_lc']:
                from matplotlib import pylab as plt
                import kplot
                kplot.plot_lc(h5)
                figpath = par['outfile'].replace('.h5','.lc.png')
                plt.gcf().savefig(figpath)
                plt.close() 

            h5.attrs['phot_plot_filename'] = os.path.basename(figpath)


def raw(h5,files,fields=[]):
    """
    Take list of .fits files and store them in the raw group

    fields - list of fields to keep. Use a subset for smaller file size.
    """
    raw  = h5.create_group('/raw')
    hduL = []
    kicL = []
    qL   = []
    for f in files:
        h = pyfits.open(f)
        hduL += [h]
        kicL += [h[0].header['KEPLERID'] ]
        qL   += [h[0].header['QUARTER'] ]

    assert np.unique(kicL).size == 1,'KEPLERID not the same'
    assert np.unique(qL).size == len(qL),'duplicate quarters'

    h5.attrs['KEPLERID'] = kicL[0] 
    for h,q in zip(hduL,qL):
        r = np.array(h[1].data)
        if fields!=[]:
            r = mlab.rec_keep_fields(r,fields)


def grid(par):
    """
    Grid Search

    Parameters
    ----------
    par : dictionary with the following keys
    
    Example
    -------

    >>> import terra
    >>> par
    {'P1': 0.5,
    'P2': 3,
    'fluxField': 'fcal',
    'fluxMask': 'fmask',
    'name': 60017806,
    'outfile': 'test.h5',
    'tbase': 7,
    'update': True}
    >>> terra.grid(dgrid)    
    
    """
    # Copy in calibrated light-curve
    names = 'P1 P2 Pcad1 Pcad2 delT1 delT2 twdG'.split()
    parL = tfind.pgramParsSeg(par['P1'],par['P2'],par['tbase'],nseg=10)
    df = pd.DataFrame(parL,columns=names)
    parL = [dict(df.ix[i]) for i in df.index]

    grid = tfind.read_hdf(par)
    grid.set_parL(parL)    
    pgram_std = grid.periodogram(mode='std')    
    grid.to_hdf(par)

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

    par = dict(par) # If passing in pandas series, treat as dict
    print "Running data_validation on %s" % par['outfile'].split('/')[-1]
    par['update'] = True  # need to use h5plus for MCMC

    outfile = par['outfile']
    PF_keys = 'LDT_deg cfrac cpad nCont'.split()
    PF_kw = dict( [ (k,par[k]) for k in PF_keys ] )

    dv = tval.DV( outfile )
    starname = h5py.File(outfile).attrs['grid_basedir'].split('/')[-1]
    dv.add_attr('starname',starname)
    dv.climb = np.array( [ par['a%i' % i] for i in range(1,5)])

    dv.at_grass()
    dv.at_SES()
    dv.at_phaseFold(0,**PF_kw)
    dv.at_phaseFold(180,**PF_kw)
    
    for ph,binsize in zip([0,0,180,180],[10,30,10,30]):
        dv.at_binPhaseFold(ph,binsize)

    dv.at_s2ncut()
    dv.at_phaseFold_SecondaryEclipse()
    dv.at_med_filt()
    dv.at_autocorr()

    trans = tm.from_dv(dv,bin_period=1)
    trans.register()
    trans.pdict = trans.fit_lightcurve()[0]
    trans.MCMC()

    # Save file and generate plot.
    dv.to_hdf(outfile,'/dv')
    trans.to_hdf(outfile,'/dv/fit')

    if par['plot_diag']:
        import tval_plotting
        from matplotlib import pylab as plt
        ext = 'pk'
        dv = tval.read_hdf(outfile,'/dv')
        dv.trans = tm.read_hdf(outfile,'/dv/fit')
        tval_plotting.diag(dv)
        figpath = par['outfile'].replace('.h5','.%s.png' % ext)
        plt.gcf().savefig(figpath)
        plt.close() 
        print "created %s" % figpath
        with h5py.File(outfile) as h5:
            h5.attrs['grid_plot_filename'] = os.path.basename(figpath)        

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


