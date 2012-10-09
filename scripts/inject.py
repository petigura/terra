import argparse
import atpy
import sim
import pandas
import glob
import keptoy
from numpy import ma
import os

parser = argparse.ArgumentParser(
    description='Inject transit into template light curve.')

parser.add_argument('inp',  type=str   , help='input folder')
parser.add_argument('out',  type=str   , help='output folder')
parser.add_argument('parfile', type=str, help='file with the transit parameters')
parser.add_argument('parrow',  type=int , help='row of the transit parameter')

args = parser.parse_args()
inp = args.inp
out = args.out
store = HDFStore('simPar.h5')
simPar = store['simPar']

d = dict(stars.ix[args.parrow])
inpfile = os.path.join(inp,d['skic']+'.h5')
outfile = os.path.join(out,d['bname']+'.h5')

os.system('cp %s %s' % (inpfile,outfile ) )
raw = prepro.Lightcurve(outfile)['raw']
for i in raw.items():
    ds = i[0]
    quarter = q[1]
    
    ft = keptoy.synMA(d,t.TIME)
    ds['f'] += ft

print "inject: Created %s"  % outfile
