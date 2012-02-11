import sys
import os

import sim
import argparse
import atpy

parser = argparse.ArgumentParser(description='Create tRES from tLC/tPAR')
parser.add_argument('RESfile',type=str)
args = parser.parse_args()

tRES = atpy.Table(args.RESfile,type='fits')
LCfile = os.path.join(os.environ['KEPSIM'],tRES.keywords['LCFILE'])
tLC  = atpy.Table(LCfile,type='fits')

VALfile = 'tVAL%04d.fits' % tLC.keywords['SEED']
dir = os.path.dirname(args.RESfile)
VALfile = os.path.join(dir,VALfile)

tVAL = sim.val(tLC,tRES,ver=False)
tVAL.keywords = tRES.keywords

tVAL.write(VALfile,type='fits',overwrite=True)
print "val: Created: %s" % VALfile
