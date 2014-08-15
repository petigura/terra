"""
Wrapper around the spline detrending.
"""
from argparse import ArgumentParser
import h5plus
import h5py
import numpy as np
from numpy import ma
import cotrend
import sys
from config import nMode,nModeSave
import stellar
import photometry
import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *


parser = ArgumentParser(description='Perform Robust SVD')
parser.add_argument('dtfile', type=str ,help='dt')
args  = parser.parse_args()

mkModeskw = dict(nMode=nMode, maxIt=4, verbose=True)

# Grab stars used for computing modes
df = stellar.read_cat()
df = df[df.prog.str.contains('GKM|cool')]
cut = df.query('11 < kepmag < 12')

epic_mode = np.array(cut.epic.tolist())

lc = photometry.read_phot(args.dtfile,epic_mode)
fdt = ma.masked_array(lc['fdt'],lc['fmask'])
U,S,V,A,fit,epic_mode_clip,fdt_clip \
    = cotrend.mkModes(fdt.copy(),epic_mode,**mkModeskw)

#out = args.dtfile.replace('dt','svd')
out = 'C0.svd.h5'


with h5plus.File(out) as h5:
    h5.create_dataset('U',data=U,compression='lzf')
    h5.create_dataset('S',data=S,compression='lzf',shuffle=True)
    h5.create_dataset('V',data=V[:nModeSave],compression='lzf',shuffle=True)
    h5.create_dataset('A',data=A[:,:nModeSave] )
    h5.create_dataset('epic',data=epic_mode_clip)

print "mkModes: created %s" % out
for i in [10,11,12,13,14]:
    cotrend.plot_modes_diag(out,i)
    pngout = out.replace('.h5','_kepmag=%i.png' % i)
    gcf().savefig(pngout)
    print "mkModes: created %s" % pngout
