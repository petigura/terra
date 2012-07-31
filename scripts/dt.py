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
parser.add_argument('out',nargs='?',type=str,help="""
output h5 file.  If not given, we just change the extention from .h5 to .dt.h5"""
                    )

args  = parser.parse_args()
inp   = args.inp
out   = h5plus.ext(args.inp,'.dt.h5',out=args.out)

h5inp = h5py.File(inp)
lc    = h5inp['LIGHTCURVE'][:]

r = prepro.modcols(lc)
r = prepro.rqmask(r)
r = prepro.rdt(r)

h5 = h5plus.File(out)
h5.create_dataset('LIGHTCURVE',data=r)

h5.close()


