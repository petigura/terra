"""
Module where all my plotting routines are found.  
Since there is some functionality common to many plotting
functions, I built a plotting class.
"""

from scipy import  stats
from scipy.optimize import leastsq,curve_fit
import numpy as np
import matplotlib.pyplot as plt
import os

import sqlite3
from uncertainties import unumpy

import postfit,getelnum
from TeX import flt2tex
from plotplus import mergeAxes,errpt,appendAxes
from numplus import binavg
from env import envset

envset(['STARSDB','PYPLOTS'])
plotdir = os.environ['PYPLOTS']
tabdir =  os.environ['TABLES']
colors = ['black','DeepSkyblue','red']
size   = [2.5,8]

class Plotgen():
    def __init__(self,plotdir=plotdir,figtype='.ps',man=False):

        ### Pull up database.
        self.conn = sqlite3.connect(os.environ['STARSDB'])
        self.cur = self.conn.cursor()    
        self.params = getelnum.Getelnum('')
        self.figtype = figtype
        self.plotdir = plotdir

        if man:
            self.figtype = '.eps'
            self.plotdir = 'ApJMS/'
            self.tabdir =  'ApJMS/'

        ### Load up single element statistics
        self.stats = {}

        for elstr in self.params.elements:
            cmd = '''
SELECT %s_abund,%s_staterrlo,%s_staterrhi 
FROM mystars 
WHERE %s ''' % (elstr,elstr,elstr,postfit.globcut(elstr))
            self.cur.execute(cmd)

            # Dump SQLite3 output into recordarray 
            temptype =[('abund',float),('staterrlo',float),('staterrhi',float)]
            arr = np.array(self.cur.fetchall(),dtype=temptype)

            m,slo,shi = np.mean(arr['abund']),np.abs(np.median(arr['staterrlo'])),np.median(arr['staterrhi'])                
            self.stats[elstr] = unumpy.uarray(([m,m],[slo,shi])) 
   
    def tfit(self,save=False,fitres=False):
        """
        Plot abundances versus temperature and overlay a cubic fit.

        save - saves the plot
        fitres - plots the residuals
        """
        figname = 'teff'
        nel = self.params.nel
        lines = self.params.lines

        f = plt.figure( figsize=(6,6) )

        ax,errx,erry = [],[],[]
        for i in range(nel):
            p = getelnum.Getelnum(lines[i])           
            elstr = p.elstr
            ax = appendAxes(ax,nel,i)
                
            fitabund, fitpar, t,abund = postfit.tfit(lines[i])    
            if fitres:
                abund = fitabund

            tarr = np.linspace(t.min(),t.max(),100)        

            p3 = np.polyfit(t,abund,3)


            ax[i].scatter(t,abund,color='black',s=10)
            ax[i].scatter(p.teff_sol,0.,color='red',s=30)

            # Plotting the best fit cubic to the temperatures
            ax[i].plot(tarr,np.polyval(p3,tarr),lw=2,color='red')        

            yerr = np.array([s.std_dev() for s in self.stats[elstr]])
            yerr = yerr.reshape((2,1))

            ax[i].set_ylabel('[%s/H]' % elstr)
            ax[i].set_xlabel('$\mathbf{ T_{eff} }$')            

            errx.append(p.tefferr)
            erry.append(yerr)

        f  = mergeAxes(f)

        for i in range(nel):
            ax[i] = errpt(ax[i],(0.95,0.9),xerr=errx[i],yerr=erry[i],
                          color = colors[2])
            ax[i].set_xlim(4500,6500)

        if save:
            plt.savefig(self.plotdir+figname+self.figtype)
            plt.close()

    def feh(self,save=False,noratio=False,texcmd=False):
        """
        Plot abundance trends against metalicty i.e. [X/Fe] against [Fe/H]

        save    - saves the file
        noratio - Plot [X/H] against [Fe/H]
        """
        #pull in fitted abundances from tfit
        figname = 'feh'
        flags  = ['dn','dk']

        bins = np.linspace(-0.3,0.5,9)

        qtype= [('abund',float),('feh',float),('staterrlo',float),
                ('staterrhi',float),('pop_flag','|S10')]    
        
        subplot = ((1,2))
        f = plt.figure( figsize=(6,6) )

        ax,errx,erry = [],[],[]
        disk = {'thick':
                    {'code':'dk',
                     'color':colors[1],
                     'ms':size[1],
                     'label':'Thick Disk',
                     'lc':colors[1],
                     },
                'thin':
                    {'code':'dn',
                     'color':colors[0],
                     'ms':size[0],
                     'label':'Thin Disk',
                     'lc':colors[2]
                     },
                }
        fd = {} # fit dictionary

        nel = self.params.nel
        lines = self.params.lines

        for i in range(nel):
            ax = appendAxes(ax,nel,i)

            p = getelnum.Getelnum(lines[i])           
            elstr = p.elstr
            fd[elstr] = {} # make a dictionary for each element

            for d in ['thin','thick']:
                cmd = '''
SELECT %s_abund,fe_abund,%s_staterrlo,%s_staterrhi,pop_flag 
FROM mystars
WHERE %s  AND pop_flag = "%s" 
''' % (elstr,elstr,elstr,postfit.globcut(elstr),disk[d]['code'])
                self.cur.execute(cmd)
                arr = np.array(self.cur.fetchall(),dtype=qtype)
            
                arr['abund'] -= p.abnd_sol
                
                x = arr['feh']                
                dy = arr['staterrlo']
                if noratio:
                    y = arr['abund']
                    ytit = '[%s/H]' % elstr
                    ax[i].set_ylabel(ytit)
                    savename = 'xonh'
                    mederrpt = (0.8,0.1)
                    axloc = 'upper left'
                else:
                    y = arr['abund'] - arr['feh']

                    ytit = '[%s/Fe]' % elstr
                    ax[i].set_ylabel(ytit)
                    savename = 'xonfe'
                    mederrpt = (0.1,0.1)
                    axloc = 'upper right'

                ax[i].plot(x,y,'o',color=disk[d]['color'],
                           ms=disk[d]['ms'],label=disk[d]['label'])


                ### Calculate best fit and save parameters ###

                # simple line to fit
                model = lambda x,m,b: m*x+b
                par,cov = curve_fit(model,x,y,p0=[1,1],sigma=dy)

                s = x.argsort()
                ax[i].plot(x[s],model(x[s],par[0],par[1]),
                           lw=3,ls='--',color=disk[d]['lc'])
                
                fd[elstr][d] = {} # make a dictionary for each stellar pop
                fitdict = fd[elstr][d]

                fitdict['quant'] = ytit

                fitdict['m'] = par[0]
                fitdict['me'] = np.sqrt(cov.diagonal()[0])

                fitdict['b'] = par[1]
                fitdict['be'] = np.sqrt(cov.diagonal()[1])

                resid = y - model(x,par[0],par[1])
                fitdict['normresid'] = resid / dy
                fitdict['chi2'] = np.sum((resid/dy)**2) / (len(x) - 2) #chi2
                

            #plot typical errorbars
            yerr = np.array([s.std_dev() for s in self.stats[elstr]])
            if not noratio:
                yerr = np.sqrt(yerr**2 + p.feherr**2)
            yerr = yerr.reshape((2,1))

            errx.append(p.feherr)
            erry.append(yerr)

            ax[i].set_xlabel('[Fe/H]')


        f = mergeAxes(f)
        leg = ax[0].legend(loc=axloc)

        for i in range(nel):
            ax[i] = errpt(ax[i],mederrpt,xerr=errx[i],yerr=erry[i],
                          color=colors[2])
        plt.show()        

        if save:
            plt.savefig(self.plotdir+savename+self.figtype)
            plt.close()
            plt.close()


            f = open(self.tabdir+'%s.tex' % savename,'w')
            line = []
            for el in fd.keys():
                for pop in fd[el].keys():
                    d = fd[el][pop]
                    line.append(\
r"""
%s & %s & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f & %.2f \\
""" % (d['quant'],pop,d['m'],d['me'],d['b'],d['be'],np.sqrt(d['chi2'])) )

            f.writelines(line)
            f.close()

        elif texcmd:
            return fd
        else:
            return fd

    def abundhist(self,save=False,texcmd=False,uplim=False):
        """
        Plot the abundance distributions.

        save - save the plot
        texcmd - returns what the tex command dumper wants

        """

        figname = 'abundhist'
        ax,nstars,outex = [],[],[]

        lines = self.params.lines
        nel = self.params.nel

        f = plt.figure( figsize=(6,6) )

        for i in range(nel):
            p = getelnum.Getelnum(lines[i])           
            elstr = p.elstr

            ax = appendAxes(ax,nel,i)

            cut = postfit.globcut(elstr,uplim=uplim)
            cmd = 'SELECT %s_abund from MYSTARS WHERE %s'% (elstr,cut)
            self.cur.execute(cmd)

            out = self.cur.fetchall()
            abund = np.array(out,dtype=float).flatten()-p.abnd_sol

            ax[i].set_ylabel('Number of Stars')
            ax[i].set_xlabel('[X/H]')
            ax[i].hist(abund,range=(-1,1),bins=20,fc='gray')

            #Annotate to show which lable we're on
            antxt = '[%s/H]' % elstr
            inv = ax[i].transData.inverted()
            txtpt = inv.transform( ax[i].transAxes.transform( (0.05,0.85) ) )
            ax[i].annotate(antxt,txtpt)

            N,m,s,min,max = abund.size,abund.mean(), \
                abund.std(),abund.min(),abund.max()
            nstars.append(N)
            if save:
            #output moments for tex write up
                outex.append(r'$ {[}%s/H] $& %i & %.2f & %.2f & %.2f & %.2f\\'
                             % (elstr,N,m,s,min,max))
            else:
                print 'N, mean, std, min, max' + antxt
                print '(%i,%f,%f,%f,%f)' % (N,m,s,min,max)
                

        f = mergeAxes(f)

        if texcmd:
            return nstars
        if save:
            plt.savefig(self.plotdir+figname+self.figtype)
            f = open(self.tabdir+'abundhist.tex','w')
            f.writelines(outex)
            plt.close()

    def comp(self,save=False,texcmd=False):
        """
        Plots my results as a function of literature
        save - saves the file
        """
        figname  = 'comp'
        ax = []
        texdict = { 'nComp':{},'StdComp':{} }

        lines = self.params.lines
        nel   = self.params.nel

        f = plt.figure( figsize=(6,8) )

        for i in range(nel):        
            p = getelnum.Getelnum(lines[i])
            elstr = p.elstr

            table = p.comptable

            if table is 'luck06':
                xfield = ',%s.c_staterr ' % table
                xcut   = ' AND %s.c_staterr < 0.3' % table
            else:
                xcut,xfield = '',''

            cmd = """
SELECT DISTINCT 
mystars.elstr_abund, mystars.elstr_staterrlo, mystars.elstr_staterrhi,
table.elstr_abund %s 
FROM 
mystars,table 
WHERE 
mystars.oid = table.oid AND table.elstr_abund IS NOT NULL AND %s %s 
""" % (xfield,postfit.globcut(elstr),xcut)
            
            cmd = cmd.replace('table',table)
            cmd = cmd.replace('elstr',elstr)
            print cmd

            self.cur.execute(cmd)
            arr = np.array(self.cur.fetchall())

            x = arr[:,3] - p.compoffset
            y = arr[:,0] - p.abnd_sol

            # Save moments into TeX shorthands
            texdict['StdComp'][elstr] = np.std(x-y)
            texdict['nComp'][elstr] = len(x)

            ###pull literature errors###
            if table is 'ben05':
                xerr = np.zeros( (2,len(x)) ) + p.comperr
            if table is 'luck06':
                xerr = arr[:,4].tolist()
                xerr = np.array([xerr,xerr])            

            yerr = np.abs(arr[:,1:3])

            # Plot the comparison
            ax.append(plt.subplot(nel,1,i+1))
            ax[i].errorbar(x,y,xerr=xerr,yerr=yerr.transpose(),
                           marker='o',ls='None',capsize=0,markersize=5,
                           color=colors[0])

            line = np.linspace(-3,3,10) # Plot 1:1 Correlation
            ax[i].plot(line,line,'k--')

            # Label and format plots
            ax[i].set_xlabel('[%s/H], %s' % (elstr,p.compref))
            ax[i].set_ylabel('[%s/H], This Work' % (elstr) )
            ax[i].set_xlim((-0.6,+0.6))
            ax[i].set_ylim((-0.6,+0.6))
            
        plt.draw()
        if texcmd:
            return texdict

        if save:
            plt.savefig(self.plotdir+figname+self.figtype)
            plt.close()

    def exo(self,save=False,prob=True,texcmd=False):
        """
        Show the fraction of stars harboring planets for different C, O 
        abundance bins
        """

        figname = 'exo'
        f = plt.figure( figsize=(6,8) )

        elements = ['O','C','Fe']
        nel = len(elements)
        sol_abnd = [8.7,8.5,0]

        ax,outex = [],[]  #empty list to store axes

        statdict = { 'host':{},'comp':{} }
        for i in range(nel): #loop over the different elements
            ax = appendAxes(ax,nel,i)

            elstr = elements[i]
            if elstr is 'Fe':
                cmd0 = 'SELECT distinct(mystars.oid),mystars.'+elstr+'_abund '+\
                    ' FROM mystars LEFT JOIN exo ON exo.oid = mystars.oid WHERE '
            else:
                cmd0 = 'SELECT distinct(mystars.oid),mystars.'+elstr+'_abund '+\
                    ' FROM mystars LEFT JOIN exo ON exo.oid = mystars.oid '+\
                    ' WHERE '+postfit.globcut(elstr)+' AND '

            #Grab Planet Sample
            cmd = cmd0 +' exo.name IS NOT NULL'
            self.cur.execute(cmd)
            #pull arrays and subtract solar offset
            arrhost = np.array( self.cur.fetchall() ,dtype=float)[:,1] - sol_abnd[i]    
            nhost = len(arrhost)

            #Grab Comparison Sample
            cmd = cmd0 +' exo.name IS NULL'
            self.cur.execute(cmd)
            #pull arrays and subtract solar offset
            arrcomp = np.array( self.cur.fetchall() ,dtype=float)[:,1] - sol_abnd[i] 
            ncomp = len(arrcomp)

            binwid = 0.1
            rng = [-0.5,0.5]
            nbins = (rng[1]-rng[0])/binwid
            if prob:
                exohist,bins = np.histogram(arrhost,bins=nbins,range=rng)
                comphist,bins = np.histogram(arrcomp,bins=nbins,range=rng)
         
                exohisterr  = unumpy.uarray( (exohist,np.sqrt(exohist)))
                comphisterr  = unumpy.uarray( (comphist,np.sqrt(comphist)))

                ratio = exohisterr/(comphisterr+exohisterr)*100.
                
                y = unumpy.nominal_values(ratio)

                yerr = unumpy.std_devs(ratio)
                x = bins[:-1]+0.5*binwid

                ax[i].hist(x,bins=bins,weights=y,color='gray')
                ax[i].errorbar(x,y,yerr=yerr,marker='o',elinewidth=2,ls='None',
                               color='black')
            else:
                # Make histogram
                ax[i].hist([arrcomp,arrhost], bins=nbins,range=rng, 
                           histtype='bar',label=['Comparison','Planet Hosts'],
                           color='gray')
                ax[i].legend(loc='upper left')

            ax[i].set_xlabel('[X/H]')
            ax[i].set_ylabel('% Stars with Planets')

            #Annotate to show which lable we're on
            inv = ax[i].transData.inverted()
            txtpt = inv.transform( ax[i].transAxes.transform( (0.05,0.85) ) )
            ax[i].annotate('[%s/H]' % elstr,txtpt)

            ######## Compute KS - statistic or probablity #########
            mhost,shost,mcomp,scomp = np.mean(arrhost),np.std(arrhost),\
                np.mean(arrcomp),np.std(arrcomp)
            D,p = stats.ks_2samp(arrhost,arrcomp)

            if save:
                # element number mean std ncomp compmean compstd KS p
                outex.append(r'{[}%s/H] & %i & %.2f & %.2f & & %i & %.2f & %.2f & %s \\' % (elstr,nhost,mhost,shost,ncomp,mcomp,scomp,flt2tex(p,sigfig=1) ) )
            else:
                print """
%s in stars w/  planets: N = %i Mean = %f Std %f 
%s in stars w/o planets: N = %i Mean = %f Std %f
KS Test: D = %f  p = %f
""" % (elstr,nhost,mhost,shost,elstr,ncomp,mcomp,scomp,D,p)

                statdict['host'][elstr] = {'n':nhost,'m':mhost,'s':shost}
                statdict['comp'][elstr] = {'n':ncomp,'m':mcomp,'s':scomp}

        f = mergeAxes(f)
        plt.draw()
        if save:
            plt.savefig(self.plotdir+figname+self.figtype)
            f = open(self.tabdir+'exo.tex','w')
            f.writelines(outex)
            plt.close()

        if texcmd:
            return statdict
        return f

    def cofe(self,save=False,texcmd=False):
        """
        Plots C/O as a function of [Fe/H]

        save - save the file
        """
        figname = 'cofe'
        o = getelnum.Getelnum('O')
        c = getelnum.Getelnum('C')
                
        cmd0 = 'SELECT distinct(mystars.oid),'+\
            ' mystars.o_abund,mystars.c_abund,mystars.fe_abund '+\
            ' FROM mystars LEFT JOIN exo ON exo.oid = mystars.oid '+\
            ' WHERE '+postfit.globcut('C')+' AND '+postfit.globcut('O')

        qtype= [('oid','|S10'),('o_abund',float),('c_abund',float),('feh',float)]

        #Grab Planet Sample
        cmd = cmd0 +' AND exo.name IS NOT NULL'
        self.cur.execute(cmd)
        #pull arrays and subtract solar offset
        arrhost = np.array( self.cur.fetchall() ,dtype=qtype)
        nhost = len(arrhost)

        #Grab Comparison Sample
        cmd = cmd0 +' AND exo.name IS NULL'
        self.cur.execute(cmd)
        #pull arrays and subtract solar offset
        arrcomp = np.array( self.cur.fetchall() ,dtype=qtype)
        ncomp = len(arrcomp)

        #calculate C/O  logeps(c) - logeps(o)
        c2ohost = 10**(arrhost['c_abund']-arrhost['o_abund'])
        c2ocomp = 10**(arrcomp['c_abund']-arrcomp['o_abund'])

        f = plt.figure( figsize=(6,4) )
        ax = plt.subplot(111)
        ax.plot(arrcomp['feh'],c2ocomp,'o',color=colors[0])
        ax.plot(arrhost['feh'],c2ohost,'o',color=colors[1])
        ax.legend(('Comparision','Hosts'),loc='upper left')
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel(r'$ \mathbf{N_C / N_O} $')



        c2o = 10**(self.stats['C']-self.stats['O'])
        yerr = np.array([s.std_dev() for s in c2o]).reshape((2,1))

        ax = errpt(ax,(0.1,0.7),xerr=o.feherr,yerr=yerr,color=colors[2])

        c2o_sol = 10**(c.abnd_sol-o.abnd_sol)
        print c2o_sol
        ax.plot([0],[c2o_sol],'o',color=colors[2],ms=size[1])

        ax.set_ybound(0,ax.get_ylim()[1])
        ax.axhline(1.,ls='--',color=colors[0])
        plt.show()

        if save:
            plt.savefig(self.plotdir+figname+self.figtype)
            plt.close()

        return f
        
    def compmany(self,elstr='o'):
        if elstr == 'o':
            tables = ['mystars','luck06','ramstars']
        if elstr =='c':
            tables = ['mystars','luck06','gus99']

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
                        cut = cut+' AND '+postfit.globcut(elstr) 

                    ax = plt.subplot(ncomp,ncomp,i*ncomp+j+1)
                    cmd = 'SELECT DISTINCT '+colx+','+coly+' FROM '+tabx+','+taby+cut
                    self.cur.execute(cmd)

                    arr = np.array(self.cur.fetchall())
                    if len(arr) > 0:
                        (x,y) = arr[:,0],arr[:,1]
                        ax.scatter(x-x.mean(),y-y.mean())

                    ax.set_xlabel(tabx)
                    ax.set_ylabel(taby)
                    xlim = ax.get_xlim()
                    x = np.linspace(xlim[0],xlim[1],10)
                    ax.plot(x,x,color='red')

        plt.draw()
        self.conn.close()


    def cooh(self,save=False,texcmd=False):
        """
        Plots C/O as a function of [Fe/H]

        save - save the file
        """

        p = getelnum.Getelnum('O')
        cmd0 = 'SELECT distinct(mystars.oid),'+\
            ' mystars.o_abund,mystars.c_abund,mystars.fe_abund,mystars.teff '+\
            ' FROM mystars LEFT JOIN exo ON exo.oid = mystars.oid '+\
            ' WHERE '+postfit.globcut('C')+' AND '+postfit.globcut('O')

        qtype= [('oid','|S10'),('o_abund',float),('c_abund',float),('feh',float),('teff',float)]

        #Thin disk
        cmd = cmd0 +' AND mystars.pop_flag IS "dn"'
        self.cur.execute(cmd)
        #pull arrays and subtract solar offset
        arrthin = np.array( self.cur.fetchall() ,dtype=qtype)


        #Grab Thick
        cmd = cmd0 +' AND mystars.pop_flag IS "dk"'
        self.cur.execute(cmd)
        #pull arrays and subtract solar offset
        arrthick = np.array( self.cur.fetchall() ,dtype=qtype)

        return arrthin,arrthick

        f = plt.figure( figsize=(6,4) )
        ax = plt.subplot(111)
        ax.plot(arrcomp['feh'],c2ocomp,'bo')
        ax.plot(arrhost['feh'],c2ohost,'go')
        ax.legend(('Comparision','Hosts'),loc='best')
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('C/O')

        c2o = 10**(self.stats['C']-self.stats['O'])
        yerr = np.array([s.std_dev() for s in c2o]).reshape((2,1))

        ax = errpt(ax,(0.1,0.7),xerr=p.feherr,yerr=yerr,color=colors[2])

        ax.set_ybound(0,ax.get_ylim()[1])
        ax.axhline(1.)
        plt.show()

        if save:
            plt.savefig(self.plotdir+figname+self.figtype)
            plt.close()

        return f

