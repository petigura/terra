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
        line.append(r'\nc{\n%s}{%i} %% # of stars in %s pop.' % (popnames[i],npop,popnames[i]))

    npoptot = np.array(npoptot).sum()
    line.append(r'\nc{\nPop}{%i} %% # of stars with pop prob' % npoptot)
    
    #Cuts specific to each line.
    
    for i in range(len(lines)):
        p = getelnum.Getelnum(el[i])
        line.append(r'\nc{\vsiniCut%s}{%i} %% vsinicut for %s' % (el[i],p.vsinicut,el[i]))

        line.append(r'\nc{\teffCut%slo}{%i} %% teff for %s' % (el[i],p.teffrng[0],el[i]))
        line.append(r'\nc{\teffCut%shi}{%i} %% teff for %s' % (el[i],p.teffrng[1],el[i]))


        fitabund,x,x,abund = postfit.tfit(lines[i])
        maxTcorr = max(np.abs(fitabund-abund))

        line.append(r'\nc{\maxT%s}{%.2f} %% max temp correction %s' % (el[i],maxTcorr,el[i]))
        line.append(r'\nc{\nComp%s}{%i} %% # of comparison stars %s' % (el[i],ncomp[i],el[i]))
        line.append(r'\nc{\StdComp%s}{%.2f} %% std of comparison stars %s' % (el[i],stdcomp[i],el[i]))
        line.append(r'\nc{\nStars%s}{%i} %% Number of stars with %s analysis' % (el[i],nstars[i],el[i]))
        

    #Values sepecific to both lines
    line.append(r'\nc{\coThresh}{%.2f} %% Theshhold for high co' % (p.coThresh))
    line.append(r'\nc{\scatterCut}{%.2f} %% Cut on the scatter' % (p.scattercut))
    line.append(r'\nc{\teffSol}{%i} %% Solar Effective Temp' % (p.teff_sol))


    statdict = plotter.exo(texcmd=True)
    for pop in statdict.keys():
        for elstr in statdict[pop].keys():
            for moment in statdict[pop][elstr].keys():
                value = statdict[pop][elstr][moment]
                if moment[0] is 'n':
                    line.append(r'\nc{\%s%s%s}{%i} %% %s - %s - %s'% 
                                (pop,elstr,moment,value,pop,elstr,moment))
                else:
                    line.append(r'\nc{\%s%s%s}{%.2f} %% %s - %s - %s'% 
                                (pop,elstr,moment,value,pop,elstr,moment))


    statdict = table.dump_stars(texcmd=True)
    for key in statdict.keys():
        value = statdict[key]
        if key[0] is 'n': #treat integers a certain way
            line.append(r'\nc{\%s}{%d} %% '% (key,value))
        else:
            line.append(r'\nc{\%s}{%.2f} %% '% (key,value))

    for l in line:        
        l = l.replace('\nc','\newcommand')
        f.write(l+'\n')

    return line
