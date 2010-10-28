import getelnum
import numpy as np
import matplotlib.pyplot as plt
import readstars
import postfit

import matchstars
from scipy.optimize import leastsq
import starsdb
import sqlite3
import scipy.stats as stats


def tfit(save=False,fitres=False):

    """
    A quick look at the fits to the temperature
    """

    line = [6300,6587]
    subplot = ((1,2))

    f = plt.figure( figsize=(6,6) )
    f.set_facecolor('white')  #we wamt the canvas to be white

    f.subplots_adjust(hspace=0.0001)
    ax1 = plt.subplot(211)
    ax1.set_xticklabels('',visible=False)
    ax1.set_yticks(np.arange(-0.8,0.4,0.2))

    ax2 = plt.subplot(212,sharex=ax1)
    ax1.set_ylabel('[O/H]')
    ax2.set_ylabel('[C/H]')
    ax2.set_xlabel('$\mathbf{ T_{eff} }$')
    ax = (ax1,ax2)

    for i in range(2):
        o = getelnum.Getelnum(line[i])           
        fitabund, fitpar, t,abund = postfit.tfit(line[i])    
        if fitres:
            abund = fitabund

        tarr = np.linspace(t.min(),t.max(),100)        
        ax[i].scatter(t,abund,color='black',s=10)
        

        ax[i].scatter(o.teff_sol,0.,color='red',s=30)
        ax[i].plot(tarr,np.polyval(fitpar,tarr),lw=2,color='red')        

    plt.xlim(4500,6500)    
    if save:
        plt.savefig('Thesis/pyplots/teff.ps')


def abundhist(save=False):

    """
    A quick look at the fits to the temperature
    """
    #pull in fitted abundances from tfit

    line = [6300,6587]
    antxt = ['[O/H]','[C/H]']

    subplot = ((1,2))
    plt.clf()
    f = plt.figure( figsize=(6,6) )
    f.set_facecolor('white')  #we wamt the canvas to be white

    f.subplots_adjust(hspace=0.0001)
    ax1 = plt.subplot(211)
    ax1.set_xticklabels('',visible=False)
    ax1.set_yticks(np.arange(0,200,50))


    
    ax2 = plt.subplot(212,sharex=ax1)
    ax2.set_yticks(np.arange(0,200,50))
    ax2.set_xlabel('[X/H]')



    ax = (ax1,ax2)


    for i in range(2):
        o = getelnum.Getelnum(line[i])           
        fitabund, fitpar, t,abund = postfit.tfit(stars,line[i])    
        ax[i].set_ylabel('Number of Stars')
        ax[i].hist(fitabund,range=(-1,1),bins=20,fc='gray')
        ax[i].set_ylim(0,200)
        ax[i].annotate(antxt[i],(-0.8,150))

        #output moments for tex write up
        print 'N, mean, std, min, max' + antxt[i]
        print '(%i,%f,%f,%f,%f)' % (fitabund.size,fitabund.mean(), \
            fitabund.luckstd(),fitabund.min(),fitabund.max())
        

    if save:
        plt.savefig('Thesis/pyplots/abundhist.ps')


def comp(save=False):
###
###  Bensby corrects his 6300 abundances for a non-LTE effect which shifts the
###  correlation away from mine by about 0.1 dex
###

    tables = [['ben05'],['luckstars']]
    offset = [[0],[8.5]]
    literr = [[0.06],[0.1]]

    lines = [6300,6587]
    color = ['blue','red','green']
    
    conn = sqlite3.connect('stars.sqlite')
    cur = conn.cursor()    

    f = plt.figure( figsize=(6,8) )
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.set_xlabel('[O/H], Bensby 2005')
    ax2.set_xlabel('[C/H], Luck 2006')

    ax1.set_ylabel('[O/H], This Work')
    ax2.set_ylabel('[C/H], This Work')
    ax = [ax1,ax2]


    for i in [0,1]:        
        xtot =[] #total array of comparison studies
        ytot =[] #total array of comparison studies

        p = getelnum.Getelnum(lines[i])
        elstr = p.elstr.lower()
        abnd_sol = p.abnd_sol

        print abnd_sol
        for j in range(len(tables[i])):
            table = tables[i][j]            

            #SELECT
            cmd = 'SELECT DISTINCT '+\
                ' mystars.'+elstr+'_abund,'+\
                ' mystars.'+elstr+'_staterrlo,'+\
                ' mystars.'+elstr+'_staterrhi,'+\
                table+'.'+elstr+'_abund'
            if table is 'luckstars':
                cmd = cmd + ','+table+'.c_staterr '

            #FROM WHERE
            cmd = cmd + \
                ' FROM mystars,'+table
            cmd = cmd + \
                ' WHERE mystars.oid = '+table+'.oid AND '+\
                table+'.'+elstr+'_abund IS NOT NULL AND '+\
                postfit.globcut(elstr)+' AND '+postfit.uplimcut(elstr)
            if table is 'luckstars':
                cmd = cmd+' AND '+table+'.c_staterr < 0.3'

            cur.execute(cmd)
            arr = np.array(cur.fetchall())
            x = arr[:,3] - offset[i][j]
            y = arr[:,0] -abnd_sol

            ###pull literature errors###
            if table is 'ben05':
                xerr = np.zeros( (2,len(x)) ) + 0.06
            if table is 'luckstars':
                xerr = arr[:,4].tolist()
                xerr = np.array([xerr,xerr])            

            yerr = np.abs(arr[:,1:3])
            print cmd
            print str(len(x)) + 'comparisons'
            
            ax[i].errorbar(x,y,xerr=xerr,yerr=yerr.transpose(),color=color[j],
                           marker='o',ls='None',capsize=0,markersize=5)
            xtot.append( x.tolist() )
            ytot.append( y.tolist() )            

        line = np.linspace(-3,3,10)

        xtot=np.array(xtot)        
        ytot=np.array(ytot)
        symerr = (yerr[:,0]+yerr[:,1])/2.

        ax[i].plot(line,line)
        ax[i].set_xlim((-0.6,+0.6))
        ax[i].set_ylim((-0.6,+0.6))

        print np.std(xtot[0]-ytot[0])
    plt.draw()
    if save:
        plt.savefig('Thesis/pyplots/comp.ps')


def exo(save=False,textab=False):
    """
    Show a histogram of Carbon and Oxygen for planet harboring stars, and
    comparison stars.
    """
    conn = sqlite3.connect('stars.sqlite')
    cur = conn.cursor()
    f = plt.figure( figsize=(6,8) )
    f.subplots_adjust(top=0.95,bottom=0.05)

    elements = ['O','C','Fe']
    nel = len(elements)
    sol_abnd = [8.7,8.5,0]

    ax = []  #empty list to store axes

    for i in range(nel): #loop over the different elements
        ax.append(plt.subplot(nel,1,i+1))
        elstr = elements[i]
        ax[i].set_xlabel('['+elstr+'/H]')
        
        if elstr is 'Fe':
            cmd0 = 'SELECT distinct(mystars.oid),mystars.'+elstr+'_abund '+\
                ' FROM mystars LEFT JOIN exo ON exo.oid = mystars.oid WHERE '
        else:
            cmd0 = 'SELECT distinct(mystars.oid),mystars.'+elstr+'_abund '+\
                ' FROM mystars LEFT JOIN exo ON exo.oid = mystars.oid '+\
                ' WHERE '+postfit.globcut(elstr)+' AND '+\
                postfit.uplimcut(elstr)+' AND '        

        #Grab Planet Sample
        cmd = cmd0 +' exo.name IS NOT NULL'
        cur.execute(cmd)
        #pull arrays and subtract solar offset
        arrhost = np.array( cur.fetchall() ,dtype=float)[:,1] - sol_abnd[i]    
        nhost = len(arrhost)

        #Grab Comparison Sample
        cmd = cmd0 +' exo.name IS NULL'
        cur.execute(cmd)
        #pull arrays and subtract solar offset
        arrcomp = np.array( cur.fetchall() ,dtype=float)[:,1] - sol_abnd[i] 
        ncomp = len(arrcomp)

        # Make histogram
        ax[i].hist([arrcomp,arrhost], 20, normed=1, histtype='bar',
                   label=['Comparison','Planet Hosts'])
        ax[i].legend(loc='upper left')

        ######## Compute KS - statistic or probablity #########
        D,p = stats.ks_2samp(arrhost,arrcomp)

        
        if textab:
            # element number mean std ncomp compmean compstd KS p
            print r'%s & %i & %.2f & %.2f & %i & %.2f & %.2f & %.2f & %f \\'\
                % (elstr,nhost,np.mean(arrhost),np.std(arrhost),
                   ncomp,np.mean(arrcomp),np.std(arrcomp),D,p)

        else:
            print elstr+' in stars w/  planets: N = %i Mean = %f Std %f ' \
                % (nhost,np.mean(arrhost),np.std(arrhost))
            print elstr+' in stars w/o planets: N = %i Mean = %f Std %f ' \
                % (ncomp,np.mean(arrcomp),np.std(arrcomp))
            print 'KS Test: D = %f  p = %f ' % (D,p)

        
    plt.draw()
    if save:
        plt.savefig('Thesis/pyplots/exo.ps')


def co(save=False):
    stars = readstars.ReadStars('Stars/keck-fit-lite.sav')

    vpasso = postfit.vbool(stars,6300)
    fitpasso = postfit.fitbool(stars,6300)
    fitpassc = postfit.fitbool(stars,6587)
    ulc = postfit.ulbool(stars,6587)
    ulo = postfit.ulbool(stars,6300)    
    ratioidx = np.where(vpasso & fitpasso & fitpassc & ~ulo & ~ulc)[0]
    ax = plt.subplot(111)
    ax.scatter(stars.o_abund[ratioidx]-8.7,stars.c_abund[ratioidx]-8.5)
    ax.set_ylabel('[C/H]')
    ax.set_xlabel('[O/H]')

    x = np.linspace(-0.8,0.6,10)
    plt.plot(x,x+0.2)
    if save:
        plt.savefig('Thesis/pyplots/co.ps')
    
def cofe(save=False):
    stars = readstars.ReadStars('Stars/keck-fit-lite.sav')

    vpasso = postfit.vbool(stars,6300)
    fitpasso = postfit.fitbool(stars,6300)
    fitpassc = postfit.fitbool(stars,6587)
    ulc = postfit.ulbool(stars,6587)
    ulo = postfit.ulbool(stars,6300)    
    ratioidx = np.where(vpasso & fitpasso & fitpassc & ~ulo & ~ulc)[0]
    ax = plt.subplot(111)
    ax.scatter(stars.feh[ratioidx],10**(stars.c_abund[ratioidx]-stars.o_abund[ratioidx]))
    ax.set_ylabel('C/O')
    ax.set_xlabel('[Fe/H]')


def compmany(elstr='o'):
    if elstr == 'o':
        tables = ['mystars','luckstars','ramstars']
    if elstr =='c':
        tables = ['mystars','luckstars','red06']

    conn = sqlite3.connect('stars.sqlite')
    cur = conn.cursor()
    ncomp = len(tables)
    for i in range(ncomp):
        for j in range(ncomp):
            if i != j:
                tabx = tables[i]
                taby = tables[j]
            
                colx = tabx+'.'+elstr+'_abund'
                coly = taby+'.'+elstr+'_abund'
                cut  = ' WHERE '+tabx+'.oid = '+taby+'.oid '+'AND '+\
                    colx+' IS NOT NULL AND '+coly+' IS NOT NULL '
                if tabx == 'mystars' or taby == 'mystars':
                    cut = cut+' AND '+postfit.uplimcut(elstr) + ' AND '+postfit.globcut(elstr) 
            
                ax = plt.subplot(ncomp,ncomp,i*ncomp+j+1)
                cmd = 'SELECT DISTINCT '+colx+','+coly+' FROM '+tabx+','+taby+cut
                cur.execute(cmd)
                    
                arr = np.array(cur.fetchall())
                if len(arr) > 0:
                    (x,y) = arr[:,0],arr[:,1]
                    ax.scatter(x-x.mean(),y-y.mean())

                ax.set_xlabel(tabx)
                ax.set_ylabel(taby)
                xlim = ax.get_xlim()
                x = np.linspace(xlim[0],xlim[1],10)
                ax.plot(x,x,color='red')

    plt.draw()
    conn.close()




