from argparse import ArgumentParser
import prepro as pp
from h5py import File as h5F
parser = ArgumentParser(description='Thin wrapper around terra module')
parser.add_argument('inp',type=str,help='raw h5file to detrend')
parser.add_argument('out',type=str,help='output h5file')

args  = parser.parse_args()
import numpy as np
np.random.seed(0)
n = 50

with h5F(args.inp) as h5raw, h5F(args.out) as h5:
    kic = h5raw['kic'][:] 
    id  = np.sort( np.random.random_integers(0,kic.size,n) )

    def getrec( i ):
        rec0 = h5raw['phot'][ i ]
        rec = pp.qdt( pp.rqmask( pp.modcols( rec0 ) ) )    
        return rec

    r0 = getrec( id[0] )
    h5.create_dataset('dt',shape=(n,r0.size), dtype=r0.dtype )

    count = 0 
    for i in id:
        h5['dt'][count] = getrec( i )
        count +=1
        if count % 10==0 : print count
