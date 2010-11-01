import table
import numpy as np
import sqlite3
import plotgen
import postfit
import getelnum

def dump(file='Thesis/texcmd.tex'):
    """
    Dumps the short hands used in the latex paper.
    """

    conn = sqlite3.connect('stars.sqlite')
    cur = conn.cursor()
    lines = [6300,6587]
    el    = ['O','C']
    f = open(file,'w')
    line = []

    line.append(r'\newcommand{\nStarsTot}{%i}' % table.dump_stars(texcmd=True))

    plotter = plotgen.Plotgen()
    ncomp = plotter.comp(texcmd=True)

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
        line.append(r'\newcommand{\n%s}{%i}' % (popnames[i],npop))

    npoptot = np.array(npoptot).sum()
    line.append(r'\newcommand{\nPop}{%i}' % npoptot)
    
    for i in range(len(lines)):
        p = getelnum.Getelnum(el[i])
        line.append(r'\newcommand{\vsiniCut%s}{%i}' % (el[i],p.vsinicut))

        fitabund,x,x,abund = postfit.tfit(lines[i])
        maxTcorr = max(np.abs(fitabund-abund))
        print maxTcorr

        line.append(r'\newcommand{\maxT%s}{%.2f}' % (el[i],maxTcorr))
        line.append(r'\newcommand{\nComp%s}{%i}' % (el[i],ncomp[i]))
    

        

    return line
