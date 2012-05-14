import sim
import argparse
import atpy
import os
import prepro

desc = """
Preprocess the input file.

read keplerio.prepLC documentation.
"""
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('inp',  type=str   , help='input file')
parser.add_argument('out',  type=str   , help='output file')
args = parser.parse_args()

tinp  = atpy.Table(args.inp)
tout  = prepro.prepLC(tinp)
tout.write(args.out,overwrite=True,type='fits')
print "prepro.py Created %s" % (args.out)
