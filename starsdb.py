"""
Builds a sqlite3 database that holds the following information:
   * Results from my spectroscopic analysis

   * Comparision results from the literature.  My sample shares some stars
     with previous studies.  Plotting literature abundances versus my own
     is a good check that my analysis is working properly.

   * Exoplanet information from exoplanets.org .  One of the goals of my
     project is to investigate whether C or O affects planet occurance rate.

I connect to the database using the amazing sqlalchemy module.  This module
lets the user define pythonic objects and then maps the attribtues of these
objects to a database.  This is a much cleaner solution than writing all the
CREATE/INSERT/UPDATE solutions.  sqlalchemy also provides a pythonic way to
*query* the database, which is much cleaner than writing out the sql commands
as a string and porting them with the sqlite3 module.  However, I started
using SQLAlchemy late in my development process, so I did not have time to
incorporate the SQLAlchemy queries.


Outline:
   * Define a `stars` SQLAlchemy object which has attributes like: name,
     teff, c_abund, o_abund, etc...

   * Define short functions that specify how to read information in from
     various tables.  Most are built upon my fxwd module.

   * Insert data into SQLAlchemy object.  The database has been created.
"""

import numpy as np
import re
import os
from matplotlib.mlab import csv2rec
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Column, Table, ForeignKey
from sqlalchemy import Integer, String, Float

from PyDL import idlobj
from fxwd import fxwd2rec
from PySIMBAD import res2id
from env import envset
#####################################
########## Reader Functions #########
#####################################

envset(['PYSTARS','STARSDB','COMP'])

def readluck06(file):
    rec = fxwd2rec(file,
                   [[0,10],[31,36],[43,48],[54,59],[65,69],[80,84],[0,0],[0,0]],
                   [('name','|S10'),('luckc505',float),
                    ('luckc538',float),('luckc659',float),
                    ('luckc514',float),('o_abund',float),
                    ('c_abund',float),('c_staterr',float)],
                   empstr='')

    for i in range(len(rec)):
        carr = np.array([ rec['luckc505'][i],rec['luckc538'][i],
                          rec['luckc659'][i],rec['luckc514'][i] ])
        carr = carr[~np.isnan(carr)]

        rec['c_abund'][i]   = np.mean(carr)
        rec['c_staterr'][i] = np.std( carr)

    return rec

def readben05(file):
    rec = fxwd2rec(file,
                   [[0,6],[293,299]],
                   [('name','|S10'),('o_abund',float)],empstr='')
    return rec

def readram07(file):
    rec = fxwd2rec(file,[[0,6],[73,78],[79,83]],
                    [('name','|S10'),('o_abund',float),('o_err',float)],
                    empstr='')
    return rec

def readben04(file):
    rec = fxwd2rec(file,[[0,6],[8,13]],
                   [('name','|S10'),('o_abund',float)],empstr='')
    return rec

def readred06(file):
    rec = fxwd2rec(file,[[17,23],[24,29],[30,35],[36,41]],
                   [('name','|S10'),('feh',float),('c_abund',float),
                    ('o_abund',float)],empstr='---')
    rec['c_abund'] += rec['feh']
    rec['o_abund'] += rec['feh']
    return rec


def readgus99(file):
    rec = csv2rec(file)
    rec['name'].dtype = np.dtype('|S10')
    return rec

compdir = os.environ['COMP']
compdict = {'luck06':
                {'reader':readluck06,
                 'simfile':compdir+'Luck06/luckresults.sim',
                 'datfile':compdir+'Luck06/Luck06py.txt',
                 },
            'ben05':
                {'reader':readben05,
                 'simfile':compdir+'Bensby05/bensby05results.sim',
                 'datfile':compdir+'Bensby05/table9.dat',
                 },
            'ram07':
                {'reader':readram07,
                 'simfile':compdir+'Ramirez07/ramirezresults.sim',
                 'datfile':compdir+'Ramirez07/ramirez.dat',
                 },
            'ben04':
                {'reader':readben04,
                 'simfile':compdir+'Bensby04/bensby04results.sim',
                 'datfile':compdir+'Bensby04/bensby04.dat',
                 },

            #This abundance has been "n-LTE corrected" meaning shifted
            #0.1 dex away from mine!
            'red06':
                {'reader':readred06,
                 'simfile':compdir+'Reddy06/reddy06results.sim',
                 'datfile':compdir+'Reddy06/table45.dat',
                 },
            'gus99':
                {'reader':readgus99,
                 'simfile':compdir+'Gustaffson99/Gustaffson99pyresults.sim',
                 'datfile':compdir+'Gustaffson99/Gustaffson99py.csv',
                 },
            }

#################################

def star_table(tabname,metadata):
    """
    Returns a generic SQLAlchemy object with the necessary fields.

    User specifies the name of the data table that appears in the sql
    database
    """

    alchobj = Table(tabname,metadata,
                    Column('id',Integer,primary_key=True),

                    ###### Stellar Information ######
                    Column('name',String(20)),
                    Column('oid',Integer),
                    Column('vsini',Float(precision=4) ),
                    Column('teff',Integer),
                    Column('pop_flag',String(10)),                    
                    Column('vmag',Float(precision=4) ),
                    Column('d',Float(precision=4) ),
                    Column('logg',Float(precision=4) ),
                    Column('monh',Float(precision=4) ),


                    ###### VF05 Work ########
                    Column('fe_abund',Float(precision=4) ),
                    Column('ni_abund',Float(precision=4) ),

                    ####### Fields Specific To Both Elements #######
                    Column('o_nfits',Float(precision=4) ),
                    Column('c_nfits',Float(precision=4) ),

                    Column('o_abund_nt',Float(precision=4) ),
                    Column('c_abund_nt',Float(precision=4) ),

                    Column('o_abund',Float(precision=4) ),
                    Column('c_abund',Float(precision=4) ),

                    Column('o_staterrlo',Float(precision=4) ),
                    Column('o_staterrhi',Float(precision=4) ),
                    Column('c_staterrlo',Float(precision=4) ),
                    Column('c_staterrhi',Float(precision=4) ),

                    Column('o_scatterlo',Float(precision=4) ),
                    Column('o_scatterhi',Float(precision=4) ),
                    Column('c_scatterlo',Float(precision=4) ),
                    Column('c_scatterhi',Float(precision=4) ),

                    # Symetric Error Fields
                    Column('o_staterr',Float(precision=4) ),
                    Column('c_staterr',Float(precision=4) ),

                    ####### Fields Specific To Both Elements #######

                    Column('o_nierrlo',Float(precision=4) ),
                    Column('o_nierrhi',Float(precision=4) ),
                    useexisting=True                
                    )
    return alchobj

def exo_table(tabname,metadata):
    alchobj = Table(tabname,metadata,
                    Column('id',Integer,primary_key=True),
                    Column('name',String(20)),
                    Column('oid',Integer),
                    Column('msini',Float(precision=4) ),
                    Column('ecc',Float(precision=4) ),
                    Column('a',Float(precision=4) ),
                    Column('per',Float(precision=4) ),
                    useexisting=True
                    )
    return alchobj

def fmtoid(idxarr,oidarr,i):
    """
    If the index i appears in the results then we include it in the sql table
    """
    if (idxarr == i).any():
        return oidarr[np.where(idxarr == i)[0][0]]
    else:
        return None

def insrec(rec):
    """
    Generates a string of 

    fieldname1=rec['fieldname1'],fieldname1=rec['fieldname2'],...
    """

    fieldnames  = list(rec.dtype.names)
    cmd = ''
    for field in fieldnames:
        if rec[field].dtype == np.dtype('|S10'):
            cmd += '%s = "%s", ' % (field,rec[field])
        else:
            cmd += '%s = %s, ' % (field,rec[field])

    cmd = cmd.replace('nan','np.nan') # Deal with the nans
    cmd = cmd[:-2] #chop off the last ' ,'
    return cmd
    
def mkdb(starsdb=None,pystars=None):
    if starsdb is None:
        starsdb = os.environ['STARSDB']
        pystars = os.environ['PYSTARS'] 

    engine = create_engine("sqlite:///"+starsdb,echo=False)
    metadata = MetaData(bind=engine)

    # if the file already exists destroy it
    if os.path.exists(starsdb):
        os.system('rm '+starsdb)

    # Declare tables.
    Mystars = star_table('mystars',metadata)
    Exo = star_table('exo',metadata)

    for key in compdict.keys():
        compdict[key]['tabob'] = star_table(key,metadata)
    

    # create tables in database
    metadata.create_all()
    conn = engine.connect()

    ### Add in Comparison data ###
    for key in compdict.keys():        
        idxarr,oidarr = res2id(compdict[key]['simfile'])
        readfunc = compdict[key]['reader']
        datfile  = compdict[key]['datfile']
        rec = readfunc(datfile)

        for i in range(len(idxarr)):
            oid = fmtoid(idxarr,oidarr,i)
            sqlobj = compdict[key]['tabob']            
            ins = sqlobj.insert()
            sqlcmd = 'obj = ins.values(%s,oid=oid)' % ( insrec(rec[i]) ) 
            exec(sqlcmd)
            conn.execute(obj)


    #### Add in mydata ####
    stars = idlobj(pystars,'stars')
    idxarr,oidarr = res2id(compdir+'myresults.sim')


    for i in range(len(stars.name)):        
        ins = Mystars.insert()
        oid = fmtoid(idxarr,oidarr,i)

        #ni abnd normalized to the sun
        ni_abund = stars.smeabund[i][27]-np.log10(stars.smeabund[i][0])+12.-6.17

        star = ins.values(name=stars.name[i],
                          oid = oid,

                          vsini = round(stars.vsini[i],3),
                          teff  = round(stars.teff[i],0),
                          pop_flag = stars.pop_flag[i],
                          vmag = round(stars.vmag[i],3),
                          d = 1/stars.prlx[i],
                          logg = round(stars.logg[i],3),
                          monh = round(stars.monh[i],3),

                          o_nfits = float(stars.o_nfits[i]),
                          c_nfits = float(stars.c_nfits[i]),
                          
                          o_abund_nt = round(stars.o_abund[i],3),
                          c_abund_nt = round(stars.c_abund[i],3),
                          
                          #place holders for the temperature corrected abundances
                          o_abund = round(stars.o_abund[i],3),
                          c_abund = round(stars.c_abund[i],3),

                          fe_abund = round(stars.feh[i],3),
                          ni_abund = round(ni_abund,3),
                          
                          o_staterrlo = round(stars.o_staterr[i,0],3),
                          o_staterrhi = round(stars.o_staterr[i,1],3),
                          
                          c_staterrlo = round(stars.c_staterr[i,0],3),
                          c_staterrhi =  round(stars.c_staterr[i,1],3),

                          o_scatterlo = round(stars.o_scatter[i,0],3),
                          o_scatterhi = round(stars.o_scatter[i,1],3),
                          
                          c_scatterlo = round(stars.c_scatter[i,0],3),
                          c_scatterhi =  round(stars.c_scatter[i,1],3),
                          
                
                          o_nierrlo = round(stars.o_nierr[i,0],3),
                          o_nierrhi = round(stars.o_nierr[i,1],3),
                          )
        conn.execute(star)

    #### Add in Exo Data ####
    idxarr,oidarr = res2id(compdir+'exoresults.sim')
    rec = csv2rec(compdir+'exoplanets-org.csv')

    for i in range(len(rec['star'])):        
        oid = fmtoid(idxarr,oidarr,i)

        ins = Exo.insert()
        exo = ins.values(name=rec['star'][i],
                         oid = oid,
                         msini = round(rec['msini'][i],3),
                         ecc   = round(rec['ecc'][i],3),
                         a     = round(rec['a'][i],3),
                         per   = round(rec['per'][i],3)
                         )            
        conn.execute(exo)
