import sim
import argparse
import atpy
import os
import keplerio

parser = argparse.ArgumentParser(
    description='Read in RAWfile and preprocess')

parser.add_argument('RAWfile',type=str)
parser.add_argument('--sim',action='store_true',help='process with CBV')

args   = parser.parse_args()
tLCraw = atpy.TableSet(args.RAWfile,type='fits')
tLC    = keplerio.prepLC(tLCraw,ver=False)

dir = os.path.dirname(args.RAWfile)
if args.sim:
    tLCrawfile = 'tLC%04d.fits' % tLC.keywords['SEED']
else:
    tLCrawfile = 'tLC%09d.fits' % tLC.keywords['KEPLERID']



tLCrawfile = os.path.join(dir,tLCrawfile)
tLC.write(tLCrawfile,overwrite=True,type='fits')
print "raw2lc: Created %s" % (tLCrawfile)
