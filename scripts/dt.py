#!/usr/bin/env python
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
import os
import atpy

dscr = """
Wrapper around detrender.

To replicate dt.py at the command line,

>>> lc = h5inp['LIGHTCURVE'][:]
>>> r = prepro.modcols(lc)
>>> r = prepro.rqmask(r)
>>> r = prepro.rdt(r)
"""

parser = ArgumentParser(description=dscr)
parser.add_argument('inp',type=str,help='input .h5 or .fits')
parser.add_argument('out',nargs='?',type=str,help="""
output h5 file.  If none, we change .h5 to .dt.h5""")

args  = parser.parse_args()


inp   = args.inp
out   = h5plus.ext(inp,'.dt.h5',out=args.out)

print inp,args.out
# Determine input type
inpext = os.path.splitext(inp)[1]
if inpext == '.fits':
    lc = atpy.Table(inp,type='fits').data
elif inpext == '.h5':
    lc = h5py.File(inp,'r+')['LIGHTCURVE'][:]    


r = prepro.modcols(lc)
r = prepro.rqmask(r)
r = prepro.rdt(r)

h5 = h5plus.File(out)
h5.create_dataset('LIGHTCURVE',data=r)
h5.close()
print "dt.py created %s" % out


