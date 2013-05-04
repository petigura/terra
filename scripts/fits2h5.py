import argparse
import glob
import pyfits
import h5py
import h5plus
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('out', type=str, help='light curve *.h5')
parser.add_argument('files',nargs='+',type=str,help='dt, cal, mqcal')
args = parser.parse_args()

out = args.out
fL = args.files

hduL = pyfits.open(fL[0])

h5 = h5plus.File(args.out)
h5.create_dataset('phot',dtype=hduL[1].data.dtype,shape=(len(fL),hduL[1].data.size))
i = 0

kic = []
for f in fL:
    hduL = pyfits.open(f)
    h5['phot'][i] = hduL[1].data
    i += 1
    if i % 100 == 0 : print i
    kic.append( hduL[0].header['KEPLERID']  )

h5['kic'] = np.array(kic)
h5.close()
