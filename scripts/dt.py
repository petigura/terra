from argparse import ArgumentParser
import prepro
from h5py import File as h5F
parser = ArgumentParser(description='Thin wrapper around terra module')
parser.add_argument('inp',type=str,help='raw file to detrend')
parser.add_argument('out',type=str,help='out file to detrend')

args  = parser.parse_args()
with h5F(args.inp) as h5raw, h5F(args.out) as h5:
    h5.copy(h5raw['raw'],'raw')

    # Perform detrending and calibration.
    prepro.mask(h5)
    prepro.dt(h5)        
    
