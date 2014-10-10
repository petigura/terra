import pandas as pd
import h5py
import sys
import sqlite3
import glob
import tval
import os
from argparse import ArgumentParser




# Table with fits column description
top_attrs="""\
#
# File paths
#
"phot_basedir","directory containing the photometry files"
"phot_fits_filename","file name of the fits file with photometry info"
"phot_plot_filename","diagnostic plots for photometry"
"grid_basedir","directory with the TERRA output files"
"grid_h5_filename","TERRA output h5 file"
"grid_plot_filename","TERRA plot file name"
"""
from cStringIO import StringIO as sio
top_attrs = sio(top_attrs)
top_attrs = pd.read_csv(top_attrs,names='field desc'.split(),comment='#')
top_attrs = top_attrs.dropna()
import tval

def dv_h5_scrape(h5file,verbose=True):
    """
    Get important MetaData and features from h5 file
    """
    d = {}

    with h5py.File(h5file,'r') as h5:
        def writekey(d,dict_key,attrs_key,cast):
            d[dict_key] = None
            try:
                d[dict_key] = cast(h5.attrs[attrs_key])
            except KeyError:
                print "KeyError %s" %attrs_key

        for k in top_attrs.field:
            writekey(d,k,k,lambda x : x)

    dv = tval.read_hdf(h5file,'/dv')
    d = dict(d,**dv.get_attrs_dict())
    return d

def dict2insert(d):
    names = str(tuple(d.keys())).replace("'","")
    strtup = "("+("?, "*len(d.keys()))[:-2]+")"
    sqlcmd = 'INSERT INTO candidate %s VALUES %s' % (names,strtup)
    values = tuple(d.values())
    return sqlcmd,values

schema = """
CREATE TABLE candidate (
  grid_basedir TEXT,
  grid_h5_filename TEXT,
  grid_plot_filename TEXT,
  id REAL,
  outfile REAL,
  phot_basedir TEXT,
  phot_fits_filename TEXT,
  phot_plot_filename TEXT,
  s2n REAL,
  s2ncut REAL,
  starname TEXT,
  t0 REAL,
  tdur REAL,
  noise REAL, 
  SES_3 REAL, 
  SES_2 REAL, 
  SES_1 REAL, 
  SES_0 REAL, 
  autor REAL, 
  s2ncut_t0 REAL, 
  SES_even REAL, 
  ph_SE REAL, 
  t0shft_SE REAL, 
  twd INT, 
  SES_odd REAL, 
  P REAL, 
  Pcad REAL, 
  grass REAL,
  s2ncut_mean REAL, 
  num_trans INT,
  mean REAL
);
"""

if __name__=='__main__':
    p = ArgumentParser(description='Scrape attributes from grid.h5 files')
    p.add_argument('h5file',type=str,nargs='+',help='h5 output file(s)')
    p.add_argument('dbfile',type=str,help='sqlite3 database file')
    args  = p.parse_args()
    
    # If table doesn't exist yet, create it.
    import os.path
    if not os.path.isfile(args.dbfile):
        con = sqlite3.connect(args.dbfile)
        cur = con.cursor()
        cur.execute(schema)
        con.commit()
        con.close()

    for h5file in args.h5file:
        d = dv_h5_scrape(args.h5file,verbose=True)
        sqlcmd,values = dict2insert(d)
        con = sqlite3.connect(args.dbfile)
        cur = con.cursor()
        cur.execute(sqlcmd,values)

    con.commit()
    con.close()
