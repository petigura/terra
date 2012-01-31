import sim
import argparse
import atpy
import os
import keplerio

parser = argparse.ArgumentParser(
    description='Read in RAWfile and preprocess')

parser.add_argument('RAWfile',type=str)
args   = parser.parse_args()
tLCraw = atpy.TableSet(args.RAWfile,type='fits')
tLC    = keplerio.prepLC(tLCraw)

dir = os.path.dirname(args.RAWfile)

tLCrawfile = 'tLC%04d.fits' % tLC.keywords['SEED']
tLCrawfile = os.path.join(dir,tLCrawfile)

tLC.write(tLCrawfile,overwrite=True)
