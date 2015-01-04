import pandas as pd
import h5py
import sys
import sqlite3
import glob
import os
from argparse import ArgumentParser
import numpy as np
from cStringIO import StringIO as sio
import transit_model as tm
import h5plus

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

top_attrs = sio(top_attrs)
top_attrs = pd.read_csv(top_attrs,names='field desc'.split(),comment='#')
top_attrs = top_attrs.dropna()

def dv_h5_scrape(h5file,verbose=True):
    """
    Get important MetaData and features from h5 file
    """
    d = {}

    with h5py.File(h5file,'r') as h5:
        def writekey(d,dict_key,attrs_key):
            d[dict_key] = None
            try:
                d[dict_key] = h5.attrs[attrs_key]
            except KeyError:
                print "KeyError %s" %attrs_key

        for k in top_attrs.field:
            writekey(d,k,k)

    def append_attrs_dict(d,h5file,groupname,prefix=''):
        try:
            io = h5plus.read_iohelper(h5file,groupname)
            d2 = io.get_attrs_dict()
            oldkeys = d2.keys()
            newkeys = [prefix+o for o in oldkeys]
            d2 = dict([(nk,d2[ok]) for nk,ok in zip(newkeys,oldkeys)])
            return dict(d,**d2)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print "%s: %s: %s" %  (groupname,exc_type,exc_value)

    d = append_attrs_dict(d,h5file,'/dv')
    d = append_attrs_dict(d,h5file,'/dv/fit',prefix='fit_')
    print d
    return d

if __name__=='__main__':
    p = ArgumentParser(description='Scrape attributes from grid.h5 files')
    p.add_argument('h5file',type=str,nargs='+',help='h5 output file(s)')
    p.add_argument('dbfile',type=str,help='sqlite3 database file')
    args  = p.parse_args()
    
    # If table doesn't exist yet, create it.
    if not os.path.isfile(args.dbfile):
        schemafile = os.path.join(
            os.environ['K2_DIR'],'code/py/candidate_schema.sql')
        with open(schemafile) as f:
            schema = f.read()

        con = sqlite3.connect(args.dbfile)
        with con:
            cur = con.cursor()
            cur.execute(schema)

    counter = 0
    for h5file in args.h5file:
        d = dv_h5_scrape(h5file,verbose=True)
        columns = ', '.join(d.keys())
        placeholders = ':'+', :'.join(d.keys())
        sql = 'INSERT INTO candidate (%s) VALUES (%s)' % \
                (columns, placeholders)

        con = sqlite3.connect(args.dbfile)
        with con:
            cur = con.cursor()
            cur.execute(sql,d)

        counter +=1
        if np.mod(counter,10)==0:
            print counter
