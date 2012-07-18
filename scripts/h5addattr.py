#!/usr/bin/env python
"""
Find the peaks in a grid.
"""
from argparse import ArgumentParser
import h5py
prsr = ArgumentParser()

prsr.add_argument('inp',type=str, help='input file')
prsr.add_argument('attr',type=str,help='Name of attribute to set')
prsr.add_argument('val',type=int,help='Value of attr')

args  = prsr.parse_args()
try:
    h5  = h5py.File(args.inp,'r+') 
    h5.attrs[args.attr]  = args.val
    h5.close()
except:
    pass

