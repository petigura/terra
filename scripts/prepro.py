import sim
import argparse
import atpy
import os
import keplerio

desc = """
Preprocess the input file.

read keplerio.prepLC documentation.
"""
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('inp',  type=str   , help='input file')
parser.add_argument('out',  type=str   , help='output file')
args = parser.parse_args()

tinp  = atpy.TableSet(args.inp)
tout  = keplerio.prepLC(tinp,ver=False)
tout.write(args.out,overwrite=True,type='fits')
print "prepro.py Created %s" % (args.out)
