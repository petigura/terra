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

import kbcUtils 
from config import G,Rsun,Rearth,Msun,AU,sec_in_day

def read_cat(cat,short=True):
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

    if cat=='kepstellar':
        cat     = '%s/keplerstellar.csv' % stellardir
        stellar = pd.read_csv(cat,skiprows=25)
        namemap = {'kepid':'kic','radius':'Rstar','prov_prim':'prov'}
        stellar = stellar.rename(columns=namemap)
    elif cat=='kic':
        cat     = '%s/kic_stellar.db' % stellardir
        con     = sqlite3.Connection(cat)
        query = 'SELECT kic,kic_teff,kic_logg,kic_radius FROM kic'
        stellar = sql.read_frame(query,con)
        namemap = {'kic_radius':'Rstar','kic_teff':'teff','kic_logg':'logg'}
        stellar = stellar.rename(columns=namemap)
        stellar = stellar.convert_objects(convert_numeric=True)
        stellar['prov'] = 'kic'
    elif cat == 'ah':
        cat     = '%s/ah_par_strip.h5' % stellardir
        store   = pd.HDFStore(cat)
        stellar = store['kic']
        namemap = {'KICID':'kic','YY_RADIUS':'Rstar','YY_LOGG':'logg',
                   'YY_TEFF':'teff'}
        stellar = stellar.rename(columns=namemap)
        stellar['prov'] = 'ah'
    elif cat == 'sm':
        cat = '/Users/petigura/Marcy/SpecMatch/sm_scrape_1423.sav'
        sm = readsav(cat)['str_arr']
        sm = pd.DataFrame(sm)
        sm.to_hdf('/var/tmp/temp.h5','temp')
        namemap = """
OBNM  obs
TEFF  teff
UTEFF uteff
LOGG  logg
ULOGG ulogg
FE    fe
UFE   ufe
M     Mstar
UM    uMstar
R     Rstar
UR    uRstar"""
        namemap = StringIO(namemap)
        namemap = pd.read_table(namemap,sep='\s*',squeeze=True,index_col=0)
        stellar = pd.read_hdf('/var/tmp/temp.h5','temp')
        stellar = stellar.rename(columns=namemap)[namemap.values]
        stellar['prov']  = 'sm'
    
        kep = kbcUtils.loadkep()[['kic','obs']].dropna()
        kep['kic'] = kep.kic.astype(int)
        stellar  = pd.merge(stellar,kep)
    else:
        print "invalid catalog"
        return None

    shortcols = 'kic teff logg prov Rstar'.split()
    if short:
        stellar = stellar[shortcols]

    stellar['Mstar'] = (stellar.Rstar*Rsun)**2 * 10**stellar.logg / G / Msun

    cat     = '%s/kic_stellar.db' % stellardir
    con     = sqlite3.Connection(cat)
    query   = 'SELECT kic,kic_kepmag FROM kic'
    mags    = sql.read_frame(query,con)
    mags    = mags.convert_objects(convert_numeric=True)
    stellar = pd.merge(stellar,mags)

    return stellar


def read_subsamp(cat):
    """
    Return a DataFrame with Kepler subsample.
    """
    if cat=='b12k':
        return pd.read_csv(stellardir+'b12k.csv')
    elif cat=='b42k':
        namemap = {'kepid':'kic'}
        cat = pd.read_csv(stellardir+'b42k.csv')
        cat = cat.rename(columns=namemap)[namemap.values()]
        return cat

def update_cat(cat1,cat2):
    """
    Default to the parameters in cat2.

    Use case: cat1 is photometrically derived parameters. cat2 is
    spectroscopically derived parameters.
    """
    cat2 = pd.merge(cat2,cat1[['kic']]) # Only use subset of stars in cat1
    cat_multi = pd.concat([cat1,cat2])  # Multiple parameters for some stars
    g = cat_multi.groupby('kic')
    cat_single = g.last() # Single parameter for each star
    cat_single['kic'] = cat_single.index
    return cat_single

def query_kic(*args):
    """         
    Query KIC
   
    Connect to Kepler kic sqlite database and return PANDAS dataframe
    """
    cnx = sqlite3.connect(os.environ['KEPBASE']+'/files/db/kic_ct.db')
    if len(args)==0:
        query  = "SELECT * FROM KIC WHERE kic=8435766"
        print "enter an sqlite query, columns include:"
        print sql.read_frame(query,cnx).T        
        return None
    
    query = args[0]
    if len(args)==2:
        kic = args[1]
        kic = tuple( list(kic) )
        query += ' WHERE kic IN %s' % str( kic ) 

    df = sql.read_frame(query,cnx)
    return df

def read_SMEatVandy(path):
    """
    Read SME-at-Vandy fits tables
    """
    
    tab = fits.open(path)[1].data
    names = tab.dtype.names

    d = {}
    for n in names:
        d[n] = tab[n][0]
    
    return pd.DataFrame(d)


