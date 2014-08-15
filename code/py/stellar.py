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

def read_cat():
    """
    Read Stellar Catalog

    Unified way to read in stellar parameters:
    
    cat : str; one of the following:
          - 'kepstellar' : Kepler stellar parameters from Exoplanet Archive
          - 'kic'        : Kepler Input Catalog
          - 'ah'         : Andrew Howard's YY corrected paramters
          - 'sm'         : SpecMatch parameters
         
    Note
    ---- 
    Mstar is a derived parameter from Rstar and logg.

    Future Work
    -----------
    Add an attribute to DataFrame returned that keeps track of the prov.
    """

    df = pd.read_csv('%s/K2_E2_targets_lc.csv' % k2_dir )
    df = df.dropna()

    namemap = dict([(c,c.strip()) for c in df.columns])
    df = df.rename(columns=namemap)
    df = df.rename(columns={'#EPIC':'epic',
                            'Kp':'kepmag',
                            'list':'prog'})
    df['prog'] = df.prog.str.slice(start=1)
    return df


def get_diag(df0,kepmag,plot_diag=False):
    """
    Return 20 stars with kepmag above certain value
    """
    df = df0.copy()
    df = df.sort('kepmag')
    df = df[df.kepmag > kepmag]

    df['ra'] = Longitude(df.ra * u.deg).wrap_at(180*u.deg).degree
    df['dec'] = Latitude(df.dec * u.deg).degree

    cent_ra,cent_dec =  354.2,-2.42
    df['dra'] = df['ra'] - median(df['ra'])
    df['ddec'] = df['dec'] - median(df['dec'])

    dfcut = df.iloc[:20]    
    if plot_diag:
        plot(df.dra,df.ddec,',')
        plot(dfcut.dra,dfcut.ddec,'.',color='Tomato',mew=0,
             label='Diagnostic Stars')
        setp(gca(),xlabel='delta RA',ylabel='delta DEC')
    return dfcut

def read_diag(kepmag):
    path_diag = '%s/Ceng/diagnostic_stars/diag_kepmag=%i.txt' % (k2_dir,kepmag)
    df = pd.read_table(path_diag,names=['epic'])
    return pd.merge(read_cat(),df)

def resolve_fits(epic):
    return "%s/Ceng/fits/kplr%09d-2014044044430_lpd-targ.fits" % (k2_dir,epic)
    
