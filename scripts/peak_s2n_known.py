#!/usr/bin/env python
"""
Add the s2n_known columns to Peak objects
"""
from argparse import ArgumentParser
import tval
import pandas
import os
import numpy as np

prsr = ArgumentParser(description='Find peaks in grid file')
prsr.add_argument('peak',type=str,nargs='?',help="Peak file")
prsr.add_argument('sim',type=str,nargs='?',help="Peak file")

args = prsr.parse_args()
peak = args.peak
simfile = args.sim
bname = os.path.basename(peak).split('.')[0]
p = tval.Peak(peak)
sim = pandas.read_table(simfile,sep='\s*')
id = np.where(sim.bname==bname)[0][0]
p.at_s2n_known(dict(sim.ix[id]) )
print "at_s2n_known: %s" % (p.attrs['skic'])
