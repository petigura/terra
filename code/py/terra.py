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
import pandas
deltaPcad = 10
from config import k2_dir

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


    with h5F(par) as h5:
        if par['type'].find('mc') != -1:
            inj(h5,par)
        
        h5.create_group('/pp')
        prepro.cal(h5,par)

        if par.has_key('plot_lc'):
            if par['plot_lc']:
                from matplotlib import pylab as plt
                import kplot
                kplot.plot_lc(h5)
                figpath = par['outfile'].replace('.h5','.lc.png')
                plt.gcf().savefig(figpath)
                plt.close() 




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
    >>> dgrid = {'P1': 0.5,'P2': 400,
                 'fluxField': 'fcal','fluxMask': 'fmask',
                 'tbase':1440,'update':True,
                 'outfile':'koi351_comp.h5'}
    >>> dgrid['P1_FFA'] = int(dgrid['P1'] / keptoy.lc)
    >>> dgrid['P2_FFA'] = int(dgrid['P2'] / keptoy.lc)
    >>> terra.grid(dgrid)    
    
    """
    names = 'P1 P2 Pcad1 Pcad2 delT1 delT2 twdG'.split()

    parL = tfind.pgramParsSeg(par['P1'],par['P2'],par['tbase'],nseg=10)
    df = pandas.DataFrame(parL,columns=names)

    print "Running grid on %s" % par['outfile'].split('/')[-1]
    print df.to_string()

    parL = [dict(df.ix[i]) for i in df.index]
    with h5F(par) as h5:     
        h5.attrs['fluxField']  = par['fluxField']
        h5.attrs['fluxMask']   = par['fluxMask']

        tfind.grid(h5,parL) 

def dv(par):
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
              'outfile':'temp.grid.h5','skic': 7831530 }


    >>> terra.dv(ddv)

    """
    par = dict(par) # If passing in pandas series, treat as dict
    print "Running dv on %s" % par['outfile'].split('/')[-1]
    par['update'] = True  # need to use h5plus for MCMC

    with h5F(par) as h5:
        if dict(h5.attrs).has_key('climb') == False:
            climb = np.array( [ par['a%i' % i] for i in range(1,5) ]  ) 
            h5.attrs['climb'] = climb

        keys = ['LDT_deg', 'cfrac', 'cpad', 'nCont']
        for k in keys:
            h5.attrs[k] = par[k]

        if dict(h5.attrs).has_key('epic') == False:
            h5.attrs['epic']  = par['epic']

        h5.noPrintRE = '.*?file|climb|epic|.*?folder'
        h5.noDiagRE  = \
            '.*?file|climb|epic|KS.|Pcad|X2.|mean_cut|.*?180|.*?folder'
        h5.noDBRE    = 'climb'

        # Attach attributes
        tval.read_dv(h5)


        tval.at_grass(h5) # How many nearby SNR peaks are there?
        tval.checkHarmh5(h5)
        tval.at_SES(h5)   # How many acceptible transits are there?

        
        if h5.attrs['num_trans'] >=2:
            tval.at_phaseFold(h5,0)
            tval.at_phaseFold(h5,180)

            tval.at_binPhaseFold(h5,0,10)
            tval.at_binPhaseFold(h5,0,30)

            tval.at_fit(h5)
            tval.at_med_filt(h5)
            tval.at_s2ncut(h5)
            tval.at_rSNR(h5)
            tval.at_autocorr(h5)

        plot_switch(h5,par)

def plot_switch(h5,par):
    ext = dict(plot_diag='pk',plot_lc='lc')
    for k in ext.keys():
        if par.has_key(k):
            if par[k]:
                from matplotlib import pylab as plt
                import kplot
                s = "kplot.%s(h5)" % k
                exec(s)
                figpath = par['outfile'].replace('.h5','.%s.png' % ext[k])
                plt.gcf().savefig(figpath)
                plt.close() 
                print "created %s" % figpath

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
    h5       : h5plus object.  Must have raw/ group
    injdict  : full pardict.  Will strip off only the necessary columns
    """
    injdict = {}
    injkeys = [k for k in pardict.keys() if k.find('inj_') !=-1 ]
    for k in injkeys:
        injdict[k[4:]] = pardict[k]

    for i in range(1,5):
        injdict['a%i' % i]=pardict['a%i' %i]
    
    raw = h5['raw']
    qL = [i[0] for i in raw.items() ]
    for q in qL:
        r  = raw[q][:] # pull data out of h5py file
        ft = keptoy.synMA(injdict,r['t'])
        r['f'] +=ft
        r = mlab.rec_append_fields(r,'finj',ft)
        del raw[q]
        raw[q] = r
        
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
    
