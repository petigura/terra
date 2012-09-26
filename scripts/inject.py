import argparse
import atpy
import sim
import pandas
import glob
import keptoy
from numpy import ma

parser = argparse.ArgumentParser(
    description='Inject transit into template light curve.')

parser.add_argument('inp',  type=str   , help='input folder')
parser.add_argument('out',  type=str   , help='output folder')
parser.add_argument('parfile', type=str, help='file with the transit parameters')
parser.add_argument('parrow',  type=int , help='row of the transit parameter')
args = parser.parse_args()
inp =args.inp
out = args.out
stars = pandas.read_table(args.parfile,sep='\s*',index_col=0)

d = dict(stars.ix[args.parrow])
d['skic'] = "%09d" % d['skic']


searchstr = inp+'Q*/kplr%s-*_llc.fits' % (d['skic'] ) 

files = glob.glob(searchstr)
for file in files:
    t = atpy.Table(file,type='fits')
    fmed = ma.masked_invalid(t.data['SAP_FLUX'])
    fmed = ma.median(fmed)

    ft = keptoy.synMA(d,t.TIME)
    ft *= fmed # Scale the flux to the median quarterly flux
    t.data['SAP_FLUX'] += ft

    outfile =file.replace(inp,out)    
    outfile = outfile[:outfile.find('kplr')]
    outfile += '%s_%06d.fits' % (d['skic'],args.parrow) 
    t.write(outfile,type='fits',overwrite=True)
    print "inject: Created %s"  % outfile
