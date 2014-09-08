"""
Module dealing with stellar parameters
"""
stellardir = '/Users/petigura/Marcy/Kepler/files/stellar/'
import os
from cStringIO import StringIO 

import sqlite3
import pandas as pd
from pandas.io import sql
from scipy.io.idl import readsav
import numpy as np
import kbcUtils 

from config import k2_dir
from config import G,Rsun,Rearth,Msun,AU,sec_in_day
from matplotlib.pylab import *

import astropy.coordinates as coord
from astropy import units as u
from astropy.coordinates import Longitude,Latitude

def read_cat(k2_camp='C0'):
    """
    Read Stellar Catalog

    Unified way to read in stellar parameters:
             
    Note
    ---- 

    Future Work
    -----------
    """

    

    if k2_camp=='Ceng':
        df = pd.read_csv('%s/catalogs/K2_E2_targets_lc.csv' % k2_dir )
        df = df.dropna()

        namemap = dict([(c,c.strip()) for c in df.columns])
        df = df.rename(columns=namemap)
        df = df.rename(columns={'#EPIC':'epic',
                                'Kp':'kepmag',
                                'list':'prog'})
        df['prog'] = df.prog.str.slice(start=1)
        ra = Longitude(df.ra*u.deg,180*u.deg)
        ra.wrap_angle=180*u.deg
        df['ra'] = ra.deg
    elif k2_camp=='C0':
        # Read in the column descriptors
        df = pd.read_table('catalogs/README_d14108_01_epic_c0_dmc',
                           header=None,names=['line'])
        df = df[df.line.str.contains('^#\d{1}')==True]
        df['col'] = df.line.apply(lambda x : x.split()[0][1:]).astype(int)
        df['name'] = df.line.apply(lambda x : x.split()[1])
        namemap = {'ID':'epic','RA':'ra','DEC':'dec','Kp':'kepmag'}

        # Read in the actual calatog
        df.index=df.name
        cut = df.ix[namemap.keys()]
        cut['newname'] = namemap.values()
        cut = cut.sort('col')
        usecols = cut.col-1
        df = pd.read_table('catalogs/d14108_01_epic_c0_dmc.mrg',
                           sep='|',names=cut.newname,header=None,
                           usecols=usecols)

    return df





def read_diag(kepmag):
    path_diag = '%s/Ceng/diagnostic_stars/diag_kepmag=%i.txt' % (k2_dir,kepmag)
    df = pd.read_table(path_diag,names=['epic'])
    return pd.merge(read_cat(),df)

def resolve_fits(epic):
    return "%s/Ceng/fits/kplr%09d-2014044044430_lpd-targ.fits" % (k2_dir,epic)
    
