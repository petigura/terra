import argparse
import glob
from numpy import rec
import h5plus
import pandas as pd
import numpy as np
import sys
from astropy.io import fits
from matplotlib import mlab

parser = argparse.ArgumentParser()
parser.add_argument('out', type=str, help='light curve *.h5')
parser.add_argument('fitsdir',type=str,help='directory with fits files')
parser.add_argument('--debug',action='store_true')
args = parser.parse_args()

out = args.out

fL  = glob.glob(args.fitsdir+'*')
if args.debug:
    fL  = np.array(fL[:100])

nfiles = len(fL)
nsample = min(nfiles,100)
np.random.seed(0)
ids   = np.sort(np.random.random_integers(0,nsample-1,nsample))
    



# Figure out the expected size of the array read 100 random
# files. Usually, I would just read in from the first file. But in the
# case, where the module failed, not all of the datasets have the same
# length
nobs  = [read_k2_fits(fL[i])[0].shape[0] for i in ids] 
df    = pd.DataFrame(nobs,columns=['len'])
group = df.groupby('len')
nobs  = group.size().idxmax()

ts0,cube0,head0,head1,head2 = read_k2_fits(fL[0])
h5 = h5plus.File(args.out)

# The image dataset stores
h5.create_dataset('cube',dtype=cube0.dtype,shape=(nfiles,nobs,50,50))
h5.create_dataset('ts',dtype=ts0.dtype,shape=(nfiles,nobs) )

i = 0
name = []
head0L = []
head1L = []
head2L = []

for f in fL:
    try:
        ts,cube,head0,head1,head2 = read_k2_fits(f)
        name += [ f.split('kplr')[1].split('-')[0] ]
        h5['cube'][i] = cube
        h5['ts'][i] = ts
        
        head0L += [head0]
        head1L += [head1]
        head2L += [head2]
    except:
        print >> sys.stderr, "problem with ", f 
    i += 1
    if i % 100 == 0 : print >> sys.stderr, i

def dict_list_to_frame(dict_list):
    df = pd.DataFrame(dict_list)
    d0 = dict( df.iloc[0] )
    goodkeys = [ k for k in d0.keys() if (type(d0[k])!=fits.card.Undefined)]
    df = df[goodkeys]

    dfs = df.select_dtypes(include=['object'])
    dfns = df.select_dtypes(exclude=['object'])

    dfs = rec.fromarrays(np.array(dfs).astype('S100').T,names=list(dfs.columns))

    names = list(dfns.columns)
    arrs = [dfns[n] for n in names]
    comb = mlab.rec_append_fields(dfs,names,arrs)
    return comb


h5['head0'] = dict_list_to_frame(head0L)
h5['head1'] = dict_list_to_frame(head1L)
h5['head2'] = dict_list_to_frame(head2L)
h5['name'] = np.array(name)

h5.close()
