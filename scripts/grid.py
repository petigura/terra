import sim
import argparse
import atpy
from numpy import ma

parser = argparse.ArgumentParser(description='Run grid search')
parser.add_argument('inp',  type=str   , help='input file')
parser.add_argument('out',  type=str   , help='output file')

args = parser.parse_args()

tLC = atpy.Table(args.inp,type='fits')
t = tLC.t
f = tLC.fdt - tLC.fcbv

fm = ma.masked_array(f,mask=tLC.fmask,fill_value=0)    
tRES = sim.grid(t,fm,tLC.isStep,Psmp = 0.25)
tRES.comments = "Table with the simulation results"
tRES.table_name = "RES"
tRES.keywords = tLC.keywords

tRES.write(args.out,type='fits',overwrite=True)
print "grid: Created %s" % args.out
