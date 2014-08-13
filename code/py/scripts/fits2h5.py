import argparse
import glob
from numpy import rec
import h5plus
import pandas as pd
import numpy as np
import sys
from astropy.io import fits


parser = argparse.ArgumentParser()
parser.add_argument('out', type=str, help='light curve *.h5')
parser.add_argument('files',nargs='+',type=str,help='dt, cal, mqcal')
args = parser.parse_args()

out = args.out
fL  = np.array(args.files)
nfiles = len(fL)


def read_k2_fits(f):
    hduL = fits.open(f)

    # Image cube. At every time step, one image. (440 x 50 x 50)
    fcube = 'RAW_CNTS FLUX FLUX_ERR FLUX_BKG FLUX_BKG_ERR COSMIC_RAYS'.split()
    cube = rec.fromarrays([hduL[1].data[f] for f in fcube],names=fcube)

    # Time series. At every time step, one scalar. (440)
    fts = 'TIME TIMECORR CADENCENO QUALITY POS_CORR1 POS_CORR2'.split()
    ts = rec.fromarrays([hduL[1].data[f] for f in fts],names=fts)

    return ts,cube


# Figure out the expected size of the array read 100 random
# files. Usually, I would just read in from the first file. But in the
# case, where the module failed, not all of the datasets have the same
# length
ids   = np.sort(np.random.random_integers(0,len(fL),100))
nobs  = [read_k2_fits(f)[0].shape[0] for f in fL[ids]] 
df    = pd.DataFrame(nobs,columns=['len'])
group = df.groupby('len')
nobs  = group.count().len.idxmax()


ts0,cube0 = read_k2_fits(fL[0])
h5 = h5plus.File(args.out)

# The image dataset stores
h5.create_dataset('cube',dtype=cube0.dtype,shape=(nfiles,nobs,50,50))
h5.create_dataset('ts',dtype=ts0.dtype,shape=(nfiles,nobs) )

i = 0
kic = []

import pdb;pdb.set_trace()
for f in fL:
    try:
    ts,cube = read_k2_fits(f)
    kic += [ f.split('kplr')[1].split('-')[0] ]
    h5['cube'][i] = cube
    h5['ts'][i] = ts
    except:
        print >> sys.stderr, "problem with ", f 
    i += 1
    if i % 100 == 0 : print >> sys.stderr, i

h5['kic'] = np.array(kic)
h5.close()
