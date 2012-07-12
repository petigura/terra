"""
Wrapper around the spline detrending.
"""

from argparse import ArgumentParser
import h5plus
import h5py
import qalg
import keplerio
import prepro
import numpy as np
from numpy import ma
import detrend
from matplotlib import mlab

parser = ArgumentParser(description='Wrapper around detrender')

parser.add_argument('inp',type=str,help='input h5 file')
parser.add_argument('out', nargs='?',type=str,help='output h5 file.  If not given, we just change the extention from .h5 to .dt.h5')
parser.add_argument('--diff',nargs='+',type=str,default=[],
                    help='list of fields to be stored individually')

args  = parser.parse_args()
inp   = args.inp
out   = args.out
if out is None:
    out = inp.replace('.h5','.dt.h5')

h5inp = h5py.File(inp)
dsinp = h5inp['LIGHTCURVE']


def func(i):
    r = dsinp[i]
    oldName = ['TIME','CADENCENO']
    newName = ['t','cad']
    for o,n in zip(oldName,newName):
        r = mlab.rec_append_fields(r,n,r[o])
        r = mlab.rec_drop_fields(r,o)

    r = keplerio.rnQ(r)
    r = keplerio.rnanTime(r)
    r = prepro.rqmask(r)
    
    # Detrend the flux
    fm = ma.masked_array(r['f'],r['fmask'])
    tm = ma.masked_array(r['t'],r['fmask'])

    # Assign a label to the segEnd segment
    label = ma.masked_array( np.zeros(r.size)-1, r['segEnd'] )

    ftnd = fm.copy()

    sL = ma.notmasked_contiguous(label)
    nseg = len(sL)
    for i in range(nseg):
        s = sL[i]
        ftnd[s]  = detrend.spldtm(tm[s],fm[s])
        label[s] = i

    r = mlab.rec_append_fields(r,'label',label.data)
    fdt   = fm-ftnd

    r = mlab.rec_append_fields(r,'ftnd',ftnd.data)
    r = mlab.rec_append_fields(r,'fdt',fdt.data)
    return r
    
t0 = func(0)
diff = ['fdt','fmask','segEnd']
h5 = h5plus.File(out)
ds,ds1d = h5plus.diffDS(dsinp.name,t0.data.dtype,dsinp.shape,h5,diff=diff)
kic = h5.create_dataset( 'KIC',data=h5inp['KIC'][:] )

ds1d[:] = mlab.rec_drop_fields(t0.data,diff)
for i in range(dsinp.shape[0]):
    if np.mod(i,100) == 0:
        print i
    r = func(i)
    ds[i] = mlab.rec_keep_fields(r,diff)


h5.close()
