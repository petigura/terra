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
args  = prsr.parse_args()

db   = args.db
grid = args.grid
out  = args.out
p = tval.Peak(out,grid,db)
p.checkHarm()
p.at_all()
print "pk %s: %s" % (p.attrs['skic'],out)
