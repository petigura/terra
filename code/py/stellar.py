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

from config import k2_dir
from config import G,Rsun,Rearth,Msun,AU,sec_in_day
from matplotlib.pylab import *

import astropy.coordinates as coord
from astropy import units as u
from astropy.coordinates import Longitude,Latitude

def read_cat(k2_camp='C0',return_targets=True):
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
        df = pd.read_table('%s/catalogs/README_d14108_01_epic_c0_dmc' % k2_dir,
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
        df = pd.read_table('%s/catalogs/d14108_01_epic_c0_dmc.mrg' % k2_dir,
                           sep='|',names=cut.newname,header=None,
                           usecols=usecols)

        df.index = df.epic

        targets = pd.read_table('%s/catalogs/C0_files.txt' % k2_dir,names=['file'])
        targets['epic'] = targets.file.apply(lambda x : x.split('/')[-1][4:13])
        targets['epic'] = targets.epic.astype(int)
        if return_targets:
            df = df.ix[targets.epic]

    return df

def read_diag(k2_camp,kepmag):
    if k2_camp=='Ceng':
        path_diag = '%s/Ceng/diagnostic_stars/diag_kepmag=%i.txt' % \
                    (k2_dir,kepmag)
    elif k2_camp=='C0':
        path_diag = '%s/catalogs/diagnostics/C0-diag_kepmag=%02d.txt' % \
                    (k2_dir,kepmag)

    df = pd.read_table(path_diag,names=['epic'])
    df = pd.merge( read_cat(k2_camp=k2_camp), df )
    return df

def resolve_fits(epic):
    return "%s/Ceng/fits/kplr%09d-2014044044430_lpd-targ.fits" % (k2_dir,epic)
    
#def query_nearby_stars(epic):
#
#
#
#pixw = max(medflux.shape) / 2
#degw = 4*pixw/3600.
#rarng = (ra-degw,ra+degw)
#decrng = (dec-degw,dec+degw)
#kepmagmax = epic.ix[202060516,'kepmag'] + 5
#
## Query all the the other targets in FOV. 
#epiccut = epic[epic.ra.between(*rarng) & epic.dec.between(*decrng) & (epic.kepmag < kepmagmax)]
#plot(xcen,ycen,'oc')
#
#text(xcen,ycen,'%(epic)09d, %(kepmag).1f' % epic.ix[202060516])
#
#coords = array(epiccut['ra dec'.split()])+np.array([dra,ddec])
#cen = w.wcs_world2pix(coords,0)
#epiccut['xcen'] = cen[:,0]
#epiccut['ycen'] = cen[:,1]
#plot(epiccut['xcen'],epiccut['ycen'],'.')
#epiccut.apply(lambda x : text(x['xcen'],x['ycen'],'%(epic)09d, %(kepmag).1f' % x),axis=1)
