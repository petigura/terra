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
parser.add_argument('out',type=str,help='output h5 file')

#parser.add_argument('n',type=str,help='output h5 file')
parser.add_argument('--diff',nargs='+',type=str,default=[],
                    help='list of fields to be stored individually')

args  = parser.parse_args()
inp   = args.inp
out   = args.out

t5 = h5py.File(inp)
rA = t5['LIGHTCURVE']

def func(i):
    r = rA[i]
    t = qalg.rec2tab(r)

    t.rename_column('TIME','t')
    t.rename_column('CADENCENO','cad')

    t = keplerio.nQ(t)
    t = keplerio.nanTime(t)
    t = prepro.qmask(t)
    
    # Detrend the flux
    fm = ma.masked_array(t.f,t.fmask)
    tm = ma.masked_array(t.t,t.fmask)
    ftnd  = detrend.spldtm(tm,fm)
    fdt   = fm-ftnd
    keplerio.update_column(t,'fdt',fdt.data)
    return t
    
t0 = func(0)
diff = ['fdt']
h5 = h5plus.File(out)
ds,ds1d = h5plus.diffDS(rA.name,t0.data.dtype,rA.shape,h5,diff=diff)
for i in range(rA.shape[0]):
    if np.mod(i,100) == 0:
        print i
    t = func(i)
    ds[i] = mlab.rec_keep_fields(t.data,diff)
h5.close()
