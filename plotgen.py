import getelnum
import numpy as np
import matplotlib.pyplot as plt
import readstars
import postfit
import matchstars
from scipy.optimize import leastsq
import starsdb
import sqlite3


def globcut(elstr):
    if elstr == 'o':
        vsinicut = '8'
    if elstr == 'c':
        vsinicut = '15'

    cut = ' mystars.vsini < '+vsinicut+\
        ' AND mystars.'+elstr+'_abund > 0  '
    return cut

def uplimcut(elstr):
    cut = ' mystars.'+elstr+'_staterrlo > -0.3 AND ' +\
    ' mystars.'+elstr+'_staterrhi < 0.3'
    return cut




def tfit(stars,save=False):

    """
    A quick look at the fits to the temperature
    """


    line = [6300,6587]
    subplot = ((1,2))
    plt.clf()
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
        fitabund, fitpar, t,abund = postfit.tfit(stars,line[i])    
        tarr = np.linspace(t.min(),t.max(),100)
        
        ax[i].scatter(t,abund,color='black',s=10)
        ax[i].scatter(o.teff_sol,0.,color='red',s=30)
        ax[i].plot(tarr,np.polyval(fitpar,tarr),lw=2,color='red')        

    plt.xlim(4500,6500)    
    if save:
        plt.savefig('Thesis/pyplots/teff.ps')


def abundhist(stars,save=False):

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
                globcut(elstr)+' AND '+uplimcut(elstr)
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
            xtot.append((x.tolist())[0])
            ytot.append((y.tolist())[0])            

        line = np.linspace(-3,3,10)

        
        ytot=np.array(ytot)
        xtot=np.array(xtot)
        symerr = (yerr[:,0]+yerr[:,1])/2.

        ax[i].plot(line,line)
        linfit = np.polyfit(xtot,ytot,1)
        ax[i].plot(line,np.polyval(linfit,line))
        ax[i].set_xlim((-0.6,+0.6))
        ax[i].set_ylim((-0.6,+0.6))

        print np.sqrt((((ytot-xtot)/symerr)**2).sum()/len(xtot-1))

    plt.draw()


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
                    cut = cut+' AND '+uplimcut(elstr) + ' AND '+globcut(elstr) 
            
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




