"""
Calibrate a bunch of LCs.
"""
from argparse import ArgumentParser
import h5plus
import h5py
from numpy import ma
import numpy as np
import keplerio
from matplotlib import mlab
import tfind

parser = ArgumentParser(description='Stictch Calibrated Light Curves Together')
parser.add_argument('out',type=str,help='output h5 file')
parser.add_argument('files',nargs='+',type=str , help='input files')

args  = parser.parse_args()
h5L = [h5py.File(f) for f in args.files]

files = args.files
nfiles = len(files)

rL = []
for j in range(nfiles):
    h5 = h5py.File(files[j],'r+')
    lc = h5['LIGHTCURVE'][:]
    rL.append(lc)
    h5.close()

rLC = keplerio.rsQ(rL)

# Add convience column for inspecting the noise on different timescales
binlen = [3,6,12]
for b in binlen:
    bcad = 2*b
    dM = tfind.mtd(rLC['t'],ma.masked_array(rLC['fcal'],rLC['fmask']),bcad)
    rLC = mlab.rec_append_fields(rLC,'dM%i' % b,dM.filled() )

h5out = h5plus.File(args.out)
h5out.create_dataset('LIGHTCURVE',data=rLC)
h5out.close()
