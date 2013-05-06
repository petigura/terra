import argparse
import glob
import pyfits
import h5py
import h5plus
import pandas as pd
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('out', type=str, help='light curve *.h5')
parser.add_argument('files',nargs='+',type=str,help='dt, cal, mqcal')
args = parser.parse_args()

out = args.out
fL  = np.array(args.files)

# Figure out the expected size of the array
ids   = np.sort(np.random.random_integers(0,len(fL),100))

nobs  = [pyfits.open(f)[1].data.shape[0] for f in fL[ids]] 
df    = pd.DataFrame(nobs,columns=['len'])
group = df.groupby('len')
nobs  = group.count().len.idxmax()


hduL = pyfits.open(fL[0])

h5 = h5plus.File(args.out)
h5.create_dataset('phot',dtype=hduL[1].data.dtype,shape=(len(fL),nobs))

i = 0
kic = []

for f in fL:
    try:
        hduL = pyfits.open(f)
        h5['phot'][i] = hduL[1].data
        kic.append( hduL[0].header['KEPLERID']  )
    except:
        print >> sys.stderr, "problem with ", f 
    i += 1
    if i % 100 == 0 : print >> sys.stderr, i

h5['kic'] = np.array(kic)
h5.close()
