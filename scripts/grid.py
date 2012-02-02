import sys
import os

import sim
import argparse
import atpy

parser = argparse.ArgumentParser(description='Create tRES from tLC/tPAR')
parser.add_argument('LCfile',type=str)
args = parser.parse_args()

tLC = atpy.Table(args.LCfile,type='fits')
tRES = sim.grid(tLC,Psmp = 0.25)
tRES.add_keyword("LCFILE" ,args.LCfile)
tRESfile = 'tRES%04d.fits' % tLC.keywords['SEED']

dir = os.path.dirname(args.LCfile)
tRESfile = os.path.join(dir,tRESfile)
tRES.write(tRESfile,type='fits',overwrite=True)
print "grid: Created %s" % tRESfile
