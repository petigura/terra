from argparse import ArgumentParser
import prepro as pp
from h5py import File as h5F
from matplotlib import mlab
import pandas as pd
import numpy as np

parser = ArgumentParser(description='Thin wrapper around terra module')
parser.add_argument('inp',type=str,help='raw h5file to detrend')
parser.add_argument('out',type=str,help='output h5file')
parser.add_argument('kic',type=str,help='.csv file with the subset of kic ids')
parser.add_argument('n',type=int,help='number stars to randomly select')

args  = parser.parse_args()
np.random.seed(0)
n = args.n

stars = pd.read_csv(args.kic,index_col=0)[['skic']]
stars = stars.rename(columns=dict(skic='kic'))

stars['subsamp'] = True

with h5F(args.inp) as h5raw, h5F(args.out) as h5:
    kic = pd.DataFrame(h5raw['kic'][:],columns=['kic'])
    kic = pd.merge(kic,stars,how='left')[['kic','subsamp']]
    kic = kic[~kic['subsamp'].isnull()]

    id = list(kic.index)
    np.random.shuffle(id)
    id = id[:n]
    id.sort()

    def getrec( i ):
        rec0 = h5raw['phot'][ i ]
        qraw = pp.rqmask( pp.modcols( rec0 ) ) 
        rec  = pp.qdt(qraw)
        rec  = mlab.rec_append_fields(rec,'fmask',qraw['fmask'])
        return rec

    r0 = getrec( id[0] )
    h5.create_dataset('dt',shape=(n,r0.size), dtype=r0.dtype )
    h5['kic'] = np.zeros(n) -1 

    count = 0 
    for i in id:
        h5['dt'][count]  = getrec( i )
        h5['kic'][count] = kic.ix[i]['kic']
        count +=1
        if count % 10==0 : print count
