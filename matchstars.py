import numpy as np
import matplotlib.mlab as mlab
from string import ascii_letters
import re
import os

import starsdb,readstars,fxwid,postfit

def simquery(code):
    """
    Generates scripted queries to SIMBAD database.  Must rerun anytime the
    ORDER of the structure changes.

    Future work:
    Automate the calls to SIMBAD.
    """

    dir = os.environ['COMP']

    datfiles = ['mystars.sim','Luck06/Luck06py.txt','exo.sim','Ramirez07/ramirez.dat','Bensby04/bensby04.dat','Bensby06/bensby06.dat','Reddy03/table1.dat','Reddy06/table45.dat','Bensby05/table9.dat','Luck05/table7.dat']

    files = ['mystars.sim','Luck06/luckstars.sim','exo.sim','Ramirez07/ramirez.sim','Bensby04/bensby.sim','Bensby06/bensby06.sim','Reddy03/reddy03.sim','Reddy06/reddy06.sim','Bensby05/bensby05.sim','Luck05/luck05.sim']


    for i in range(len(files)):
        files[i] = dir+files[i]
        datfiles[i] = dir+datfiles[i]

    if code == 0:
        stars = readstars.ReadStars(os.environ['PYSTARS'])
        names = stars.name
        simline = names2sim(names,cat='HD')
    if code == 1:
        names, c, o = postfit.readluck(datfiles[code])
        simline = names2sim(names,cat='')
    if code == 2:
        rec = mlab.csv2rec('smefiles/exoplanets-org.csv')
        names = rec['simbadname']
        simline = names2sim(names,cat='')
    if code == 3:
        names,a,a,a = starsdb.readramirez(datfiles[code])
        simline = names2sim(names,cat='HIP')

    if code == 4:
        names,o = starsdb.readbensby(datfiles[code])
        simline = names2sim(names,cat='HIP')

    if code == 5:
        names,c = starsdb.readbensby06(datfiles[code])
        simline = names2sim(names,cat='HD')

    if code == 6:
        names = (fxwid.rdfixwid(datfiles[code],[[0,6]],['|S10']))[0]
        simline = names2sim(names,cat='HD')

    if code == 7:
        names = (fxwid.rdfixwid(datfiles[code],[[17,23]],['|S10']))[0]
        simline = names2sim(names,cat='HIP')

    if code == 8:
        names = (fxwid.rdfixwid(datfiles[code],[[0,6]],['|S10']))[0]
        simline = names2sim(names,cat='HIP')

    if code == 9:
        names,x,x = starsdb.readluck05(datfiles[code])
        simline = names2sim(names,cat='HD')


    f = open(files[code],'w')
    f.writelines(simline)
    f.close()

def names2sim(names,cat=''):
    """
    Creates a SIMBAD scripted query from from an of star names.

    cat - SIMBAD must know what catalog the star is from. If names already
          specify the catalog, we're done.  If not set cat to append the
          catalog to the names file.
          
          HD  - Henry-Draper Catalog
          HIP - Hiparcos Catalog

    >>> names = np.array(['14412','4915'])
    >>> sim = names2sim(names,cat='HD')

    To use in a SIMBAD script, one must write to a file
    --------------File Output--------------
    result oid

    echodata -n x0
    query id  14412

    echodata -n x1
    query id  4915
    ---------------------------------------

    Future work:
    Make unit tests.
    """

    query   = np.array([],dtype='|S50')

    #Add whitespace between catalog and star name
    if cat is not '':
        cat += ' '

    #First line of query - specifies we want oid output
    query = np.append(query,'result oid\n')

    for i in np.arange( len(names) ):
        name = names[i]
        name = cat+name

        # Two lines of SIMBAD script per star.  The 'x' is a necessary
        # placehold for subsequent parsing
        queryline = 'echodata -n x%i\nquery id %s\n' % (i,name)
        query = np.append(query,queryline) 

    return query

def res2id(file):
    """
    Parses SIMBAD query results into matched pairs of (idx,oid)

    Query Results have this form:
    
    :: Header :::::::
    script snipet
    console
    error messages

    :: data :::::::::

    x0#1: 1356191
    x1#1: 1225912
    x2#1: 759900
    x3#1: 936604

    res2id chops off the header and returns    
    0, 1356191
    1, 1225912
    2, 759900
    3, 936604

    Future Work:
    Create unit test.
    """

    idxarr,oidarr = np.array([]),np.array([])

    f = open(file,'r')
    lines = f.readlines()
    for i in np.arange(len(lines)):
        if re.search('#1',lines[i]) is not None:
            idx,sep,oid = lines[i].partition('#1: ')
            idxarr = np.append(idxarr,int(idx[idx.rfind('x')+1:]))
            oidarr = np.append(oidarr,int(oid[:-1]))

    return idxarr,oidarr
