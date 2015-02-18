#!/usr/bin/env python
"""
Effectively a wrapper around tval.pkInfo()
"""
from argparse import ArgumentParser
import tval

prsr = ArgumentParser(description='Find peaks in grid file')
prsr.add_argument('out',type=str,nargs='?', help="output file")
prsr.add_argument('grid',type=str, help=".grid.h5 file")
prsr.add_argument('db',type=str, help='Database with limb-darkening coeffs')
prsr.add_argument('--debug',type=bool,default=False,help='Shorter run')
args  = prsr.parse_args()

db   = args.db
grid = args.grid
out  = args.out

p = tval.Peak(out,grid,db)
p.checkHarm()
p.at_phaseFold()
p.at_fit()
p.at_med_filt()
p.at_s2ncut()
p.at_SES()
p.at_rSNR()
p.at_autocorr()

print "pk %s: %s" % (p.attrs['skic'],out)
