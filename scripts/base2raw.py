import sim
import argparse
import atpy
import os

parser = argparse.ArgumentParser(
    description='Read in basefile inject a transit')

parser.add_argument('PARfile',type=str)
parser.add_argument('seed',type=int)
args = parser.parse_args()

PARfile = args.PARfile
PARfile = os.path.abspath(PARfile)

tPAR = atpy.Table(PARfile,type='fits')
tPAR = tPAR.where(tPAR.seed == args.seed)
KIC = tPAR.KIC
dir = os.path.dirname(PARfile)
BASEfile = os.path.join(dir,'tBASE_%09d.fits' % KIC)

tLCbase = atpy.TableSet(BASEfile,type='fits')
tLCraw = sim.inject(tLCbase,tPAR)

tLCraw.keywords['PARFILE'] = os.path.relpath( PARfile,os.environ['KEPSIM'] )
tLCrawfile = 'tLCraw%04d.fits' % args.seed
tLCrawfile = os.path.join(dir,tLCrawfile)
tLCraw.write(tLCrawfile,overwrite=True,type='fits')
print "base2raw: Created %s" % tLCrawfile
