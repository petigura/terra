import numpy as np
import matplotlib.mlab as mlab
from string import ascii_letters
import re
import os

from fxwd import fxwd2rec
import starsdb,postfit
from PyDL import idlobj

def simquery(code):
    """
    Generates scripted queries to SIMBAD database.  Must rerun anytime the
    ORDER of the initial stars structure changes.

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
        stars = PyDL.idlobj(os.environ['PYSTARS'],'stars')
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
        names = fxwd2rec(datfiles[code],[[0,6]],['|S10']) ['name']
        simline = names2sim(names,cat='HD')

    if code == 7:
        names = fxwd2rec( datfiles[code],[[17,23]],['|S10'] ))['name']
        simline = names2sim(names,cat='HIP')

    if code == 8:
        names = fxwd2rec( datfiles[code],[[0,6]],['|S10'] )['name']
        simline = names2sim(names,cat='HIP')

    if code == 9:
        names,x,x = starsdb.readluck05(datfiles[code])
        simline = names2sim(names,cat='HD')


    f = open(files[code],'w')
    f.writelines(simline)
    f.close()

