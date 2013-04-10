"""
TERRA

The top level controller.  This function encapsulates the complete pipeline.
"""

import h5py
from h5py import File as h5F 

import numpy as np

import prepro
import tfind
import tval
import keptoy
import config
from matplotlib import mlab
import h5plus
import copy
import os

deltaPcad = 10

nessflds = {
'tps':'skic,id,sid,a1,a2,a3,a4',
'mcL':'skic,id,sid,a1,a2,a3,a4,inj_P,inj_tau,inj_p,inj_phase,inj_b',
'mcS':'skic,id,sid,a1,a2,a3,a4,inj_P,inj_tau,inj_p,inj_phase,inj_b,gridfile',
}


#######################
# Top Level Functions #
#######################

def pp(par):
    """
    Preprocess

    Parameters
    ----------
    par : dictionary with the following keys
          - rawfile 
          - outfile
          - type = mc/tps
          
          - inj_P,inj_phase,inj_p,inj_tau,inj_b      <-- only for mc
          - a1,a2,a3,a4 limb darkening coefficients  <-- inj/rec runs

    Example
    -------

    >>> import terra
    >>> dpp = {'a1': 0.77, 'a2': -0.67, 'a3': 1.14,'a4': -0.41,
    'inj_P': 142.035,'inj_b': 0.46,'inj_p': 0.0132, 
    'inj_phase': 0.583,'inj_tau': 0.178,
    'outfile': 'temp.grid.h5','rawfile': 'lc/007831530.h5',
    'skic': 7831530, 'svd_folder': '../terra5-50/svd/', 'type': 'mc'}
    >>> terra.pp(dpp)
    
    """

    with h5F(par['rawfile']) as h5raw, h5F(par['outfile']) as h5:

        h5.copy(h5raw['raw'],'raw')
        if par['type'].find('mc') != -1:
            inj(h5,par)

        # Perform detrending and calibration.
        prepro.mask(h5)
        prepro.dt(h5)        
        prepro.cal(h5,par['svd_folder'])
        prepro.sQ(h5)

def grid(par):
    """
    Grid Search

    Parameters
    ----------
    par : dictionary with the following keys
    
    Example
    -------

    >>> import terra
    >>> dgrid = {'P1_FFA': 2000,'P2_FFA': 2200,
    'fluxField': 'fcal','fluxMask': 'fmask',
    'outfile' : 'temp.grid.h5'}
    >>> terra.grid(dgrid)    

    """

    with h5F(par['outfile']) as h5:     
        h5.attrs['fluxField']  = par['fluxField']
        h5.attrs['fluxMask']   = par['fluxMask']

        for k in 'P1_FFA,P2_FFA'.split(','):
            h5.attrs[k] = par[k]

        tfind.grid(h5) 
        tfind.itOutRej(h5)

def dv(par):
    """
    Data Validation
    
    Parameters
    ----------
    par : dictionary with the following keys

    Example
    -------

    >>> import terra
    >>> ddv {'LDT_deg': 3,'cfrac': 3, 'cpad': 0.5, 'nCont': 4,
    'a1': 0.773,'a2': -0.679,'a3': 1.140, 'a4': -0.416,
    'outfile': 'temp.grid.h5','skic': 7831530}    
    >>> terra.dv(ddv)

    """

    with h5F(par['outfile']) as h5:
        if dict(h5.attrs).has_key('climb') == False:
            climb = np.array( [ par['a%i' % i] for i in range(1,5) ]  ) 
            h5.attrs['climb'] = climb

        keys = ['LDT_deg', 'cfrac', 'cpad', 'nCont']
        for k in keys:
            h5.attrs[k] = par[k]

        if dict(h5.attrs).has_key('skic') == False:
            h5.attrs['skic']  = par['skic']

        h5.noPrintRE = '.*?file|climb|skic|.*?folder'
        h5.noDiagRE  = \
            '.*?file|climb|skic|KS.|Pcad|X2.|mean_cut|.*?180|.*?folder'
        h5.noDBRE    = 'climb'

        # Attach attributes
        h5.RES,h5.lc = get_reslc(h5)

        tval.findPeak(h5)
        tval.conv(h5)
        tval.checkHarmh5(h5)
        tval.at_phaseFold(h5,0)
        tval.at_phaseFold(h5,180)

        tval.at_binPhaseFold(h5,0,10)
        fitgrp = h5.create_group('fit')

        tval.at_fit(h5,h5['blc10PF0'],fitgrp,runmcmc=True)
        tval.at_med_filt(h5)
        tval.at_s2ncut(h5)
        tval.at_SES(h5)
        tval.at_rSNR(h5)
        tval.at_autocorr(h5)


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

def get_reslc(h5):
    """
    Get res array.

    3 Cases
    - Peak.  it0/RES, mqcal
    - No peak, no outliers. it0/mqcal
    - No peak, outliers itN/fmask
    """
    try:
        lc  = h5['mqcal'][:]
        itMax = findItMax(h5)
        lc['fmask'] = h5[itMax]['fmask']
        RES = h5[itMax]['RES'][:]
    except KeyError:
        lc  = h5['mqcal'][:]
        RES = h5[itMax]['RES'][:]
        
    return RES,lc

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
    
def makescripts(df):
    df['rawfile']    = 'eb10k_slim/lc/'+df.skic+'.h5'
    df['svd_folder'] = 'eb10k/svd/'

    type = raw_input("run type [tps/mcL/mcS] : ")
    df['type'] = type

    if type !='tps':
        labels = ['tau','p','b','phase','P']
        for k in labels:
            df['inj_'+k] = df[k]
        df = df.drop(labels,axis=1)

    for f in nessflds[type].split(','):
        assert dict(df.ix[0]).has_key(f)

    basedir = raw_input("%s " % os.environ['KEPSCRATCH'])

    for case in ['test','full']:
        casedir=basedir+case+'/'
        print casedir
        os.makedirs(casedir)

        dirs = {}
        for s in ['pngs','h5','csv','csvout']:
            dirs[s] = casedir+s+'/'
            os.makedirs(dirs[s])

        df['pngGrid']  = dirs['pngs']+df.sid+'.png'
        df['storeGrid']= dirs['h5']+df.sid+'.grid.h5'

        if case=='test':
            dfS  = df.ix[:15]
            dfS['P1']        = 2180
            dfS['P2']        = 2200
            dfL = np.array_split(dfS,16)
        elif case=='full':
            nfiles = 1000 
            dfL = np.array_split(df,nfiles)

        for i in range(len(dfL)):
            dfL[i].to_csv(dirs['csv']+'test-%04d.csv' % i)
    return df
