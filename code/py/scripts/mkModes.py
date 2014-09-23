"""
Wrapper around the spline detrending.
"""
from argparse import ArgumentParser
from glob import glob
import os

from astropy.io import fits
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *

import photometry
import h5plus
import prepro
from cotrend import EnsembleCalibrator,makeplots
import k2_catalogs
import cPickle as pickle


parser = ArgumentParser(description='Ensemble calibration')
parser.add_argument('fitsdir',type=str,help='directory with fits files')
parser.add_argument('-f',action='store_true',
                     help='Force re-creation of h5 checkpoint file')
args = parser.parse_args()

# Loading up all the fits files takes some time. The first order of
# business is to create a checkpoint.

dir = args.fitsdir
dirname = os.path.dirname(dir)
h5file = "%s.h5" % (dirname)

if not os.path.exists(h5file) or args.f:
    fL = glob('%s/*.fits' % dirname)
    nfL = len(fL)
    print "loading up %i files" % nfL

    lcL = []
    epicL = []

    for i in range(nfL):
        f = fL[i]
        lc = photometry.read_crossfield_fits(fL[i])
        lc = prepro.rdt(lc)
        epic = os.path.basename(f).replace('.fits','')
        if i%100==0:
            print i

        lcL+=[lc]
        epicL+=[epic]

    lc = np.vstack(lcL)
    epic = np.array(epicL)
    
    print "Creating h5 checkpoint: %s" % h5file
    with h5plus.File(h5file) as h5:
        h5['lc'] = lc
        h5['epic'] = epic

# Load up files from h5 database
print "Reading files from %s" % h5file
with h5py.File(h5file) as h5:
    lc = h5['lc'][:]
    epic = h5['epic'][:]

fdt = ma.masked_array(lc['fdt'],lc['fmask'])
epic = epic.astype(int)
dftr = pd.DataFrame(index=epic)
dftr['epic'] = dftr.index
targets = k2_catalogs.read_cat(return_targets=True)
dftr = pd.merge(dftr,targets.drop_duplicates())

import pdb;pdb.set_trace()    
ec = EnsembleCalibrator(fdt,dftr)
ec.robust_components(algo='PCA')

ec.plot_basename = ec.plot_basename.replace('cotrend',dirname)
makeplots(ec,savefig=True)

picklefn = dirname+'_ec.pickle'
print "Saving calibrator object to %s" % picklefn
with open(picklefn,'w') as f:
    pickle.dump(ec,f)
