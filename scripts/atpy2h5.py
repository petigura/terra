import argparse
import atpy
import sim
import h5py
import glob
import numpy as np
import os

parser = argparse.ArgumentParser(
    description='Inject transit into template light curve.')

parser.add_argument('inp',  type=str   , 
                    help='input files passed to glob')

parser.add_argument('out',  type=str   , 
                    help='output h5 file')

# Unpack arguments
args  = parser.parse_args()
inp   = args.inp
out   = args.out

# Must write into a new h5 file
if os.path.exists(out):
    os.remove(out)

files = glob.glob(inp)
f     = h5py.File(out)
nfiles= len(files)

csize     = 300e3 # Target size uncompressed size for the chunks.
ccolsize  = min(100,nfiles)

# Array Data Type
t0    = atpy.Table(files[0],type='fits')
arrdtype = t0.data.dtype

# Compute chunksize.
elsize = arrdtype.itemsize
crowsize  = int(csize/elsize/ccolsize)
chunks = (ccolsize,crowsize)

print "Creating Dataset with (%i,%i)" % chunks

ds = f.create_dataset(t0.table_name,(nfiles,t0.data.size),arrdtype,chunks=chunks,compression='lzf',shuffle=True)

kwL = []
for i in range(nfiles):
    tfile = files[i]
    if np.mod(i,100)==0:
        print i
    t = atpy.Table(tfile,type='fits')

    # Copy the data over
    ds[i] =  t.data

    # Construct keyword dictionary
    kwd = t.keywords
    kwd['file'] = tfile
    kwL.append(kwd)

for k in t0.keywords.keys:
    ds.attrs[k] = np.array([kw[k] for kw in kwL])

f.close()
