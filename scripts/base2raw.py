import sim
import argparse
import atpy
import os

parser = argparse.ArgumentParser(
    description='Read in basefile inject a transit')

parser.add_argument('BASEfile',type=str)
parser.add_argument('PARfile',type=str)
parser.add_argument('seed',type=int)
args = parser.parse_args()
tLCbase = atpy.TableSet(args.BASEfile)
tPAR = atpy.Table(args.PARfile)

tLCraw = sim.inject(tLCbase,tPAR,args.seed)

tLCraw.keywords['PARFILE'] = args.PARfile
dir = os.path.dirname(args.BASEfile)
tLCrawfile = 'tLCraw%04d.fits' % args.seed
tLCrawfile = os.path.join(dir,tLCrawfile)
tLCraw.write(tLCrawfile,overwrite=True)
