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
    """
    if pardict['type']=='mcS':
        assert pardict.has_key('gridfile'),'must point to full res file'
        
    name = "".join(np.random.random_integers(0,9,size=10).astype(str))    
    with h5py.File( pardict['rawfile'],'r+' ) as h5raw, \
         prepro.Lightcurve(name+'.h5',driver='core',backing_store=False) as h5out,\
         tval.Peak(name+'.pk.h5',driver='core',backing_store=False) as p:

        # Raw photometry
        h5out.copy(h5raw['raw'],'raw')
        h5raw.close()
        if pardict['type'].find('mc') != -1:
            inj(h5out,pardict)

        # Perform detrending and calibration.
        h5out.mask()
        h5out.dt()
        h5out.attrs['svd_folder'] = pardict['svd_folder']
        h5out.cal()
        h5out.sQ()

        # Grid Search and iterative outlier rejection
#        h5out.attrs['P1'] = 1000
#        h5out.attrs['P2'] = 1050
        h5out.attrs['P1'] = int(config.P1 / config.lc)
        h5out.attrs['P2'] = int(config.P2 / config.lc)

        if pardict['type'] == 'mcS':
            gridShort(h5out,pardict)
        else:
            tfind.grid(h5out) 

        tfind.itOutRej(h5out)

        if pardict.has_key('storeGrid'):
            h5store = h5plus.File(pardict['storeGrid'],mode='c')
            h5store['RES']   = h5out['RES'][:]
            h5store['mqcal'] = h5out['mqcal'][:]
            h5store.close()

        # DV
        p['RES']   = h5out['RES'][:]
        p['mqcal'] = h5out['mqcal'][:]

        climb = np.array( [ pardict['a%i' % i] for i in range(1,5) ]  ) 
        p.attrs['climb'] = climb
        p.attrs['skic']  = pardict['skic']

        p.findPeak()
        p.conv()
        p.checkHarm()
        p.at_phaseFold()
        p.at_fit()
        p.at_med_filt()
        p.at_s2ncut()
        p.at_SES()
        p.at_rSNR()
        p.at_autocorr()
        if pardict.has_key('pngGrid'):
            import matplotlib
            matplotlib.use('Agg')
            import kplot
            from matplotlib.pylab import plt
            kplot.plot_diag(p)
            plt.gcf().savefig(pardict['pngGrid'])

        out = p.flatten(p.noDBRE)
        out['id'] = pardict['id']
        h5out.close()
        p.close()


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
