import sys
import os

import sim
import argparse
import atpy
from numpy import ma

parser = argparse.ArgumentParser(description='Create tRES from tLC/tPAR')
parser.add_argument('LCfile',type=str)
parser.add_argument('--cbv',action='store_true',help='process with CBV')
args = parser.parse_args()

tLC = atpy.Table(args.LCfile,type='fits')
t = tLC.t

if args.cbv:
    f = tLC.fdt - tLC.fcbv
else:
    f = tLC.f

fm = ma.masked_array(f,mask=tLC.fmask,fill_value=0)    
tRES = sim.grid(t,fm,Psmp = 0.25)
tRES.comments = "Table with the simulation results"
tRES.table_name = "RES"
tRES.keywords = tLC.keywords
tRES.keywords['LCFILE'] = os.path.relpath(args.LCfile,os.environ['KEPSIM'] )

tRESfile = 'tRES%04d.fits' % tLC.keywords['SEED']

dir = os.path.dirname(args.LCfile)
tRESfile = os.path.join(dir,tRESfile)
tRES.write(tRESfile,type='fits',overwrite=True)
print "grid: Created %s" % tRESfile
