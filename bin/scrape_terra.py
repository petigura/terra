#!/usr/bin/env python
import os
import sqlite3
import glob
from argparse import ArgumentParser
from cStringIO import StringIO as sio

import pandas as pd
import h5py
import numpy as np

from terra.scrape_terra import create_table, dv_h5_scrape, insert_dict

if __name__=='__main__':
    p = ArgumentParser(description='Scrape attributes from grid.h5 files')
    p.add_argument('h5file',type=str,nargs='+',help='h5 output file(s)')
    p.add_argument('dbfile',type=str,help='sqlite3 database file')
    args  = p.parse_args()
    
    # If table doesn't exist yet, create it.
    create_table(args.dbfile)

    counter = 0
    for h5file in args.h5file:
        d = dv_h5_scrape(h5file,verbose=True)
        insert_dict(d,args.dbfile)
        counter +=1
        if np.mod(counter,10)==0:
            print counter
