#!/usr/bin/env python
"""
Effectively a wrapper around tval.pkInfo()

"""
from argparse import ArgumentParser
import h5py
import h5plus
import tval
import numpy as np
import keptoy
import sqlite3
import os
import logging


prsr = ArgumentParser(description='Find peaks in grid file')
prsr.add_argument('cal',type=str, help=".cal.h5 file")
prsr.add_argument('grid',type=str, help=".grid.h5 file")
prsr.add_argument('db',type=str, help='Database with limb-darkening coeffs')
prsr.add_argument('out',type=str,nargs='?', help="output file.  If None, we replace .grid.h5 --> .pk.h5 ")
args  = prsr.parse_args()

cal  = args.cal
db   = args.db
grid = args.grid
out  = args.out

p = tval.Peak(cal,grid,db)
p.phaseFold()
p.fit()
p.s2ncut()
p.write(out,overwrite=True)
print "pk %s: %s" % (p.attrs['skic'],out)
