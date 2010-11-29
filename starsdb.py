# To do Monday ...
# Transfer over the star table function
# Replace the class definitions with calls to star_table
# See if the OID logic can be tightened (put in its own function)
# Define an exoplanet object
# Make sure the throughly test the pipeline 

import matplotlib
import numpy as np
import re
import os
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Column, Table, ForeignKey
from sqlalchemy import Integer, String, Float

import readstars,fxwid
from matchstars import res2id

def readluck05(file):
    names,el,abund,err = \
        fxwid.rdfixwid(file,[[0,6],[37,42],[43,48],[49,53]],
                       ['|S10','|S10',np.float,np.float])
    idx = (np.where(el == 'C I  '))[0]

    stop
    c_abund = abund[idx]
    c_staterr = err[idx]
    names = names[idx]

    return names,c_abund,c_staterr


def readramirez(file):
    """
    Reads in file from ramirez.dat
    """
    f = open(file,'r')
    txt = f.readlines()
    nstars = len(txt)

    star = np.zeros(nstars,dtype='|S10') #hiparcos name 
    teff = np.zeros(nstars,dtype=np.float) #effective temp
    o = np.zeros(nstars,dtype=np.float) #oxygen abudnance n-LTE corrected
    o_err = np.zeros(nstars,dtype=np.float) #oxygen abundance error n-LTE

    for i in np.arange(nstars):
        line = txt[i]
        star[i],teff[i],o[i],o_err[i] = \
            line[0:6],line[14:18],line[73:78],line[79:83]

    return star,teff,o,o_err


def readbensby06(file):
    """
    Reads in file from bensby06.dat
    """
    f = open(file,'r')
    txt = f.readlines()
    nstars = len(txt)

    star = np.zeros(nstars,dtype='|S10') #HD name 
    c_abund = np.zeros(nstars,dtype=np.float) #oxygen abudnance n-LTE corrected

    for i in np.arange(nstars):
        line = txt[i]        
        line = line.split('\t')

        star[i],c_abund[i]= line[1],line[3]

    return star,c_abund

def star_table(tabname,metadata):
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


                    ####### Fields Specific To Both Elements #######

                    Column('o_nierrlo',Float(precision=4) ),
                    Column('o_nierrhi',Float(precision=4) ),
                
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

def mkdb():
#    metadata.bind.echo = True
    engine = create_engine("sqlite:///"+os.environ['STARSDB'],echo=True)
    metadata = MetaData(bind=engine)
    

    # if the file already exists destroy it
    if os.path.exists(os.environ['STARSDB']):
        os.system('rm '+os.environ['STARSDB'])

    # Declare tables.
    Luckstars = star_table('luckstars',metadata)
    Ramstars =  star_table('ramstars',metadata)
    Ben04 =  star_table('ben04',metadata)
    Ben05 =  star_table('ben05',metadata)
    Ben06 =  star_table('ben06',metadata)
    Red03 =  star_table('red03',metadata)
    Red06 =  star_table('red06',metadata)

    # create tables in database
    metadata.create_all()
    conn = engine.connect()

    #### Add in mydata ####
    stars = readstars.ReadStars(os.environ['PYSTARS'])
    idxarr,oidarr = res2id('Comparison/myresults.sim')




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

    #### Add in Luck Data ####
    luckname,luckc505,luckc538,luckc659,luckc514,lucko = \
        fxwid.rdfixwid('Comparison/Luck06/Luck06py.txt',
                       [[0,10],[31,36],[43,48],[54,59],[65,69],[80,84]],
                       ['|S10',np.float,np.float,np.float,np.float,np.float],
                       empstr='')

    idxarr,oidarr = res2id('Comparison/Luck06/luckresults.sim')

    for i in range(len(luckname)):        
        oid = fmtoid(idxarr,oidarr,i)
        ins = Luckstars.insert()

        carr = np.array([luckc505[i],luckc538[i],luckc659[i],luckc514[i]])
        cavg = np.mean(carr)
        cstd = np.std(carr)

        star = ins.values(name=luckname[i],
                          oid = oid,
                          o_abund = round(lucko[i],3),
                          c_abund = round(cavg,3),
                          c_staterr = round(cstd,3)
                          )

    #### Add in Exo Data ####
    idxarr,oidarr = res2id('Comparison/exoresults.sim')
    rec = matplotlib.mlab.csv2rec('Comparison/exoplanets-org.csv')
    for i in range(len(rec['star'])):        
        if (idxarr == i).any():
            oid = str(oidarr[np.where(idxarr == i)[0][0]])

        else:
            oid = None

        Exo(name=rec['star'][i],
            oid = oid,
            msini = round(rec['msini'][i],3),
            ecc   = round(rec['ecc'][i],3),
            a     = round(rec['a'][i],3),
            per   = round(rec['per'][i],3)
            )            


    #### Add in Ramirez Data ####
    idxarr,oidarr = res2id('Comparison/Ramirez07/ramirezresults.sim')
    names,teff,o,o_err = readramirez('Comparison/Ramirez07/ramirez.dat')

    for i in range(len(names)):        
        if (idxarr == i).any():
            oid = str(oidarr[np.where(idxarr == i)[0][0]])
        else:
            oid = None

        Ramstars(name=names[i],
                  oid = oid,
                  teff = round(teff[i],0),
                  o_abund = round(o[i],3),
                  o_err = round(o_err[i],3)
                  )

    #### Add in Bensby Data ####
    idxarr,oidarr = res2id('Comparison/Bensby04/bensby04results.sim')
    #This abundance has been "n-LTE corrected" meaning shifted 0.1 dex away from
    #mine!
    names,o_abund = fxwid.rdfixwid('Comparison/Bensby04/bensby04.dat',
                                   [[0,6],[8,13]],['|S10',np.float],empstr='')


    for i in range(len(names)):        
        if (idxarr == i).any():
            oid = str(oidarr[np.where(idxarr == i)[0][0]])
        else:
            oid = None

        Ben04(name=names[i],
              oid = oid,
              o_abund = round(o_abund[i],3),
              )
        

    idxarr,oidarr = res2id('Comparison/Bensby06/bensby06results.sim')
    names,c_abund = readbensby06('Comparison/Bensby06/bensby06.dat')

    for i in range(len(names)):        
        if (idxarr == i).any():
            oid = str(oidarr[np.where(idxarr == i)[0][0]])
        else:
            oid = None

        Ben06(name=names[i],
              oid = oid,
              c_abund = round(c_abund[i],3),
              )
        

    idxarr,oidarr = res2id('Comparison/Reddy03/reddy03results.sim')
    names,feh = fxwid.rdfixwid('Comparison/Reddy03/table1.dat',
                                  [[0,6],[17,22]],['|S10',np.float])
    
    names,c_abund,o_abund = fxwid.rdfixwid('Comparison/Reddy03/table5.dat',
                               [[0,6],[7,12],[19,23]],
                               ['|S10',np.float,np.float],empstr='----')
    

    for i in range(len(names)):        
        if (idxarr == i).any():
            oid = str(oidarr[np.where(idxarr == i)[0][0]])
        else:
            oid = None

        Red03(name=names[i],
              oid = oid,
              feh = round(feh[i],3),
              c_abund = round(c_abund[i]+feh[i],3),
              o_abund = round(o_abund[i]+feh[i],3),
              )
        




    idxarr,oidarr = res2id('Comparison/Reddy06/reddy06results.sim')    
    names,feh,c_abund,o_abund = fxwid.rdfixwid('Comparison/Reddy06/table45.dat',
                               [[17,23],[24,29],[30,35],[36,41]],
                               ['|S10',np.float,np.float,np.float],empstr='---')
    

    for i in range(len(names)):        
        if (idxarr == i).any():
            oid = str(oidarr[np.where(idxarr == i)[0][0]])
        else:
            oid = None

        Red06(name=names[i],
              oid = oid,
              c_abund = round(c_abund[i]+feh[i],3),
              o_abund = round(o_abund[i]+feh[i],3),
              )


    idxarr,oidarr = res2id('Comparison/Bensby05/bensby05results.sim')    
    names,o_abund = fxwid.rdfixwid('Comparison/Bensby05/table9.dat',
                               [[0,6],[293,299]],
                               ['|S10',np.float],empstr='')
    

    for i in range(len(names)):        
        if (idxarr == i).any():
            oid = str(oidarr[np.where(idxarr == i)[0][0]])
        else:
            oid = None

        Ben05(name=names[i],
              oid = oid,
              o_abund = round(o_abund[i],3),
              )
        

    session.commit()
    session.close()
