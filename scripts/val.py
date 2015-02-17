import sim
import argparse
import atpy

parser = argparse.ArgumentParser(description='Run light curve validation')

parser.add_argument('lcfile'  ,  type=str, help='light curve file')
parser.add_argument('gridfile',  type=str, help='grid result file')
parser.add_argument('out'     ,  type=str, help='output file')

args = parser.parse_args()

tRES = atpy.Table(args.gridfile,type='fits')
tLC  = atpy.Table(args.lcfile  ,type='fits')
tVAL = sim.val(tLC,tRES,ver=False)
tVAL.keywords = tRES.keywords
tVAL.write(args.out,type='fits',overwrite=True)
print "val: Created: %s" % args.out
