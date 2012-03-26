import sim
import argparse
import atpy

parser = argparse.ArgumentParser(description='Run light curve validation')

parser.add_argument('val'  , type=str, help='light curve file')
args = parser.parse_args()

sim.bfitval(args.val)

tVAL.write(args.out,type='fits',overwrite=True)
print "val: Created: %s" % args.out
