import argparse
import atpy
import sim
import h5py
import glob
import numpy as np

parser = argparse.ArgumentParser(
    description='Inject transit into template light curve.')

parser.add_argument('inp',  type=str   , 
                    help='input files passed to glob')

parser.add_argument('out',  type=str   , 
                    help='output h5 file')

args  = parser.parse_args()
files = glob.glob(args.inp)
f     = h5py.File(args.out)
nfiles= len(files)

# Array Data Type
t0    = atpy.Table(files[0],type='fits')
arrdtype = t0.data.dtype

# KW dtype
names = t0.keywords.keys
types = [type(v) for v in t0.keywords.values]

kwdtype = [(n,t) for n,t in zip(names,types) if t!=type('str')]
kwdtype = np.dtype(kwdtype)

arrds = f.create_dataset(t0.table_name,(nfiles,t0.data.size),arrdtype,
                         compression='lzf',chunks=(nfiles,100),shuffle=True)

kwds  = f.create_dataset("kw",(nfiles,),kwdtype,compression='lzf')

for i in range(nfiles):
    t = atpy.Table(files[i],type='fits')

    # Copy the data over
    arrds[i] =  t.data

    # And the keywords
    for n in kwdtype.names:
        kwds[i][n]  = t.keywords[n]
