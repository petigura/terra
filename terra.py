"""
TERRA

The top level controller.  This function encapsulates the complete pipeline.
"""

import h5py
import numpy as np

import prepro
import tfind
import tval
import keptoy
import config
from matplotlib import mlab
import h5plus

def terra(pardict):
    """
    
    Parameters
    ----------
    
    pardict : All the attributes are controlled through this dictionary.
    out     : Dictionary of scalars. Must contain id key.
    
    """
    if pardict['type']=='mcS':
        assert pardict.has_key('gridfile'),'must point to full res file'
        
    name = "".join(np.random.random_integers(0,9,size=10).astype(str))    
    with h5py.File( pardict['rawfile'],'r+' ) as h5raw, \
         prepro.Lightcurve(name+'.h5',driver='core',backing_store=False) as h5out:

        # Raw photometry
        h5out.copy(h5raw['raw'],'raw')
        if pardict['type'].find('mc') != -1:
            inj(h5out,pardict)

        # Perform detrending and calibration.
        h5out.mask()
        h5out.dt()
        h5out.attrs['svd_folder'] = pardict['svd_folder']
        h5out.cal()
        h5out.sQ()

        # Grid Search and iterative outlier rejection
        h5out.attrs['P1'] = int(config.P1 / config.lc)
        h5out.attrs['P2'] = int(config.P2 / config.lc)
        if pardict.has_key('P1'):
            h5out.attrs['P1'] = pardict['P1']
            h5out.attrs['P2'] = pardict['P2']

        if pardict['type'] == 'mcS':
            gridShort(h5out,pardict)
        else:
            tfind.grid(h5out) 

        tfind.itOutRej(h5out)

        # DV
        DV(h5out,pardict)

        if pardict.has_key('pngGrid'):
            import matplotlib
            matplotlib.use('Agg')
            import kplot
            from matplotlib.pylab import plt
            kplot.plot_diag(h5out)
            plt.gcf().savefig(pardict['pngGrid'])

        if pardict.has_key('storeGrid'):
            with h5plus.File(pardict['storeGrid'],mode='c') as h5store:
                h5store.copy(h5out,h5store,name='store')

        out = tval.flatten(h5out,h5out.noDBRE)
        out['id'] = pardict['id']
    return out

def PP(h5,pardict):
    """
    Preprocessing

    Parameters
    ----------
    h5 : h5plus object


    """


def DV(h5,pardict):
    """
    Perforn Data Validation

    Parameters
    ----------
    h5 : h5plus object

    """
    
    climb = np.array( [ pardict['a%i' % i] for i in range(1,5) ]  ) 
    h5.attrs['climb'] = climb
    h5.attrs['skic']  = pardict['skic']

    h5.noPrintRE = '.*?file|climb|skic|.*?folder'
    h5.noDiagRE  = '.*?file|climb|skic|KS.|Pcad|X2.|mean_cut|.*?180|.*?folder'
    h5.noDBRE    = 'climb'

    tval.findPeak(h5)
    tval.conv(h5)
    tval.checkHarmh5(h5)
    tval.at_phaseFold(h5)
    tval.at_fit(h5)
    tval.at_med_filt(h5)
    tval.at_s2ncut(h5)
    tval.at_SES(h5)
    tval.at_rSNR(h5)
    tval.at_autocorr(h5)

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
        
def addPaths(df):
    """
    Paths are referenced with respect to a base directory, which can
    changed from machine to machine.
    """
    for column in ['storeGrid','pngGrid','gridfile','svd_folder','rawfile']:
        df[column] = df['wkdir'] + df[column]


def gridShort(h5,pardict):
    """

    """
    with h5py.File(pardict['gridfile'],'r+') as grid0:
    # Only compute the grid over a narrow region in period
        res0 = grid0['RES'][:]
        
    deltaPcad = 10
    P1=int(pardict['inj_P']/config.lc - deltaPcad)
    P2=int(pardict['inj_P']/config.lc + deltaPcad)
    h5.attrs['P1'] = max(P1 , res0['Pcad'][0] )
    h5.attrs['P2'] = min(P2 , res0['Pcad'][-1] )

    tfind.grid(h5)

    # Add that narrow region to the already computed grid.
    res  = h5['RES'][:]
    start = np.where(res0['Pcad']==res['Pcad'][0])[0]
    stop  = np.where(res0['Pcad']==res['Pcad'][-1])[0]
    if len(start) > 1:
        start = start[1]
    if len(stop) > 1:
        stop = stop[0]
    res0[start:stop+1] = res
    del h5['RES']
    h5['RES'] = res0
