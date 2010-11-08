import table
import numpy as np
import sqlite3
import plotgen
import postfit
import getelnum
import os

def dump(file='Thesis/texcmd.tex'):
    """
    Dumps the short hands used in the latex paper.
    """

    conn = sqlite3.connect(os.environ['STARSDB'])
    cur = conn.cursor()
    lines = [6300,6587]
    el    = ['O','C']
    f = open(file,'w')
    line = []

    line.append(r'\newcommand{\nStarsTot}{%i} %% Total # of Stars Analyzed (C & O) ' % table.dump_stars(texcmd=True))

    plotter = plotgen.Plotgen()
    ncomp,stdcomp = plotter.comp(texcmd=True)
    nstars= plotter.abundhist(texcmd=True)

    #number of different disk populations

    popnames = ['Thin','Thick','Halo','Bdr']
    popsym   = ['dn','dk','h','dn/dk']
    npoptot  = []

    for i in range(len(popnames)):
        cmd = """
SELECT count(pop_flag) FROM mystars WHERE pop_flag = "%s" 
AND
((%s) OR (%s))
""" % (popsym[i],postfit.globcut('O'),postfit.globcut('C') )

        print cmd
        cur.execute(cmd)
        npop = (cur.fetchall())[0][0]
        npoptot.append(npop)
        line.append(r'\newcommand{\n%s}{%i} %% # of stars in %s pop.' % (popnames[i],npop,popnames[i]))

    npoptot = np.array(npoptot).sum()
    line.append(r'\newcommand{\nPop}{%i} %% # of stars with pop prob' % npoptot)
    
    for i in range(len(lines)):
        p = getelnum.Getelnum(el[i])
        line.append(r'\newcommand{\vsiniCut%s}{%i} %% vsinicut for %s' % (el[i],p.vsinicut,el[i]))

        fitabund,x,x,abund = postfit.tfit(lines[i])
        maxTcorr = max(np.abs(fitabund-abund))

        line.append(r'\newcommand{\maxT%s}{%.2f} %% max temp correction %s' % (el[i],maxTcorr,el[i]))
        line.append(r'\newcommand{\nComp%s}{%i} %% # of comparison stars %s' % (el[i],ncomp[i],el[i]))
        line.append(r'\newcommand{\StdComp%s}{%.2f} %% std of comparison stars %s' % (el[i],stdcomp[i],el[i]))
        line.append(r'\newcommand{\nStars%s}{%i} %% Number of stars with %s analysis' % (el[i],nstars[i],el[i]))
        

    for l in line:
        
        f.write(l+'\n')
    return line
