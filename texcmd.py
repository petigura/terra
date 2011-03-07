import numpy as np
import os

import table,plotgen,postfit,getelnum
from env import envset

envset(['PAPER'])

def dump()
    """
    Dumps the shorthands used in the latex paper.

    Any statistic derived from my analysis that I quote in my paper should
    be generated automatically.  This function generates a set of TeX
    shorthands for mean abundance, number of stars analyzed, etc.
    """

    plotter = plotgen.Plotgen()
    cur = plotter.cur

    params = getelnum.Getelnum('')
    lines = params.lines
    el    = params.elements

    f = open(file,'w')
    line = []


    nstars = plotter.abundhist(texcmd=True)

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

        line.append(r'\nc{\nStars%s}{%i} %% Number of stars with %s analysis' % (el[i],nstars[i],el[i]))
        

    #Values sepecific to both lines
    line.append(r'\nc{\coThresh}{%.2f} %% Theshhold for high co' % (p.coThresh))
    line.append(r'\nc{\scatterCut}{%.2f} %% Cut on the scatter' % (p.scattercut))
    line.append(r'\nc{\teffSol}{%i} %% Solar Effective Temp' % (p.teff_sol))

    compdict = plotter.comp(texcmd=True)


    # Add in keywords from comparison study
    for moment in compdict.keys():
        for elstr in compdict[moment].keys():
            value = compdict[moment][elstr]
            if moment[0] is 'n':
                line.append(r'\nc{\%s%s}{%i}   ' % (moment,elstr,value) )
            else:
                line.append(r'\nc{\%s%s}{%.2f} ' % (moment,elstr,value) )

    # Add in keywords from exoplanet study
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

    # Add in keywords from table generation
    statdict = table.dump_stars(texcmd=True)
    for key in statdict.keys():
        value = statdict[key]
        if key[0] is 'n': #treat integers a certain way
            line.append(r'\nc{\%s}{%d} %% '% (key,value))
        else:
            line.append(r'\nc{\%s}{%.2f} %% '% (key,value))



    for l in line:        
        l = l.replace('\nc','\newcommand')

    return line
