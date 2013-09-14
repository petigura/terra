import pandas as pd
import h5py
import tval
from argparse import ArgumentParser as ArgPar
import sys

parser = ArgPar(description='determine expected terra SNR for given koi')
parser.add_argument('koi',type=str,help='koi')
args  = parser.parse_args()

q12 = analysis.loadKOI('Q12')
q12.index = q12.koi
kic = q12.ix[args.koi,'kic'] # find lookup star number

grid = pd.read_csv('grid.csv',index_col=0)
grid['kic'] = grid.outfile.apply(lambda x : x.split('/')[-1][:9]).astype(int)
grid.index = grid.kic
par = grid.ix[kic]
s2n = analysis.kois2n(args.koi,par)
print koi,s2n
