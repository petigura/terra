"""
Wrapper around the spline detrending.
"""

from argparse import ArgumentParser
import h5py

parser = ArgumentParser(description='Wrapper around detrender')

parser.add_argument('inp',type=str,help='input h5 file')
parser.add_argument('out',type=str,help='input h5 file')
parser.add_argument('dsname',type=str,help='output h5 file')

args  = parser.parse_args()
inp   = args.inp
out   = args.out
dsname= args.dsname
h5inp = h5py.File(inp)
h5out = h5py.File(out)
h5out.create_dataset(dsname,data=h5inp[dsname][:])
h5out.close()
