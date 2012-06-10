import sim
import h5py
import glob
import numpy as np
import os
from matplotlib import mlab
import atpy

def compChunks(elsize,ncolmax):
   """
   Compute Chunck Size
   """

   # Compute chunksize.
   csize     = 300e3 # Target size uncompressed size for the chunks.
   crowsize = 100
   ccolsize  = int(csize/elsize/crowsize)
   ccolsize  = min(ccolsize,ncolmax)
   chunks = (ccolsize,crowsize)
   return chunks

def atpy2h5(inp,out,diff=[]):
   """
   atpy format to h5

   Parameters
   ----------

   inp  : globable string specifying where the input files are
   out  : output h5 file.  In none exists, we create it.
   diff : List of fields that are different to be stored as stacked arrays.
          If the fields are the same, we store as a 1-d array

   """

   # Must write into a new h5 file
   if os.path.exists(out):
       os.remove(out)

   files = glob.glob(inp)
   files = np.array(files)

   f     = h5py.File(out)
   nfiles= len(files)

   # Array Data Type
   t0    = atpy.Table(files[0],type='fits')
   arrdtype = t0.data.dtype

   # Store the fields that are the same in table_name1d

   if diff != []:
      ds1dname = t0.table_name+'1d'
      ds1data  = t0.data
      ds1data  = mlab.rec_drop_fields(ds1data,diff)

      ds1d     = f.create_dataset(ds1dname,data=ds1data)

      same = list(arrdtype.names)
      [same.remove(d) for d in diff]
      r = mlab.rec_drop_fields(t0.data,same)

      print """
Same Columns
------------"""
      print "%s " % ', '.join(map(str,same))

      print """
Diff Columns
------------"""
      print "%s " % ', '.join(map(str,diff))

   else:
      r = t0.data
      same = []

   chunks = compChunks(r.dtype.itemsize,nfiles )
   ccolsize,crowsize = chunks

   ncol = nfiles
   nrow = t0.data.size

   ds = f.create_dataset(t0.table_name,(ncol,nrow),r.dtype,
                         chunks=chunks,compression='lzf',shuffle=True)

   print "ds.shape = (%i,%i); ds.chunks=(%i,%i)" % (ds.shape + ds.chunks)

   kwL = []
   start = 0
   stop = 0

   while stop < nfiles:
       stop = start + ccolsize
       stop = min(stop,nfiles)

       print stop
       s = slice(start,stop)

       rL = []
       kwLtemp = []

       for tf in files[s]:
          t = atpy.Table(tf,type='fits')
          r = mlab.rec_drop_fields(t.data,same)
          rL.append(r)

          # Pull out the dictionaries
          d = t.keywords
          d['file'] = tf
          kwLtemp.append(d)

       ds[s] =  np.vstack(rL)
       kwL = kwL + kwLtemp

       start = stop
   
   print """
Attaching Keywords
------------------"""
   print "%s " % ', '.join(map(str,kwL[0].keys))

   keys = []
   for k in kwL[0].keys:
      tk = type(kwL[0][k])
      if (tk is str) or (tk is bool) or (tk is float):      
         keys.append(k)
         ds.attrs[k] = np.zeros(nfiles,dtype=tk)
   

   for k in keys:
      for i in range(nfiles):
         try:
            ds.attrs[k][i] = kwL[i][k]
         except:
            pass
   f.close()




































