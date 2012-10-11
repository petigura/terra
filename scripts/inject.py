import argparse
import keptoy
import pandas
import os
import prepro

parser = argparse.ArgumentParser(
    description='Inject transit into template light curve.')

parser.add_argument('inp',  type=str   , help='input folder')
parser.add_argument('out',  type=str   , help='output folder')
parser.add_argument('parfile', type=str, help='file with the transit parameters')
parser.add_argument('parrow',  type=int , help='row of the transit parameter')

args = parser.parse_args()
inp = args.inp
out = args.out
simPar = pandas.read_csv(args.parfile,index_col=0)
d = dict(simPar.ix[args.parrow])

inpfile = os.path.join(inp,"%09d.h5" % d['skic'])
outfile = os.path.join(out,d['bname']+'.h5')

os.system('cp %s %s' % (inpfile,outfile ) )
raw = prepro.Lightcurve(outfile)['raw']
for i in raw.items():
    quarter = i[0]
    ds = i[1]
    r = ds[:]
    ft = keptoy.synMA(d,r['t'])
    r['f'] += ft
    ds[:] = r



print "inject: Created %s"  % outfile
