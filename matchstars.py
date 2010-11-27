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
    make simbad script from stars names.
    
    implicit_hd kw assumes if the string is just a number it is from the HD
    catalog.
    """
    nstars = len(names)
    simnames  = np.zeros(nstars,dtype='|S20')
    simline =  np.zeros(nstars+1,dtype='|S50')
    simline[0] = 'result oid\n'

    for i in np.arange(nstars):
        name = names[i]
        if re.search(name[0],ascii_letters) is None:
            simnames[i] = cat+' '+name
        else:
            simnames[i] = name

        placeholder = 'x'+str(i)+'\n'
        simline[i+1] = 'echodata -n '+placeholder+'query id '+simnames[i]+'\n'

    return simline

def res2id(file):
    """
    convert simbad query result into list simbad id
    """
    f = open(file,'r')
    lines = f.readlines()
    idxarr =[]
    oidarr = []

    for i in np.arange(len(lines)):
        if re.search('#1',lines[i]) is not None:
            idx,sep,oid = lines[i].partition('#1: ')
            idxarr.append(int(idx[idx.rfind('x')+1:]))
            oidarr.append(int(oid[:-1]))

    idxarr = np.array(idxarr)
    oidarr = np.array(oidarr)
    return idxarr,oidarr
