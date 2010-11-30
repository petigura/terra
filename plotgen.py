from scipy import  stats
from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
import os

import sqlite3
from uncertainties import unumpy

import readstars,postfit,matchstars,starsdb,getelnum,flt2tex
from plotplus import mergeAxes,errpt,appendAxes

class Plotgen():
    def __init__(self):
    ###Pull up database.
        self.conn = sqlite3.connect(os.environ['STARSDB'])
        self.cur = self.conn.cursor()    
        self.params = getelnum.Getelnum('')
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
        A quick look at the fits to the temperature
        save - saves the plot
        fitres - plots the residuals
        """
        nel = self.params.nel
        lines = self.params.lines

        f = plt.figure( figsize=(6,6) )

        ax = []
        for i in range(nel):
            p = getelnum.Getelnum(lines[i])           
            elstr = p.elstr
            ax = appendAxes(ax,nel,i)
                
            fitabund, fitpar, t,abund = postfit.tfit(lines[i])    
            if fitres:
                abund = fitabund

            tarr = np.linspace(t.min(),t.max(),100)        

            ax[i].scatter(t,abund,color='black',s=10)
            ax[i].scatter(p.teff_sol,0.,color='red',s=30)
            ax[i].plot(tarr,np.polyval(fitpar,tarr),lw=2,color='red')        

            yerr = np.array([s.std_dev() for s in self.stats[elstr]])
            yerr = yerr.reshape((2,1))

            ax[i] = errpt(ax[i],(0.9,0.9),xerr=p.tefferr,yerr=yerr)

            ax[i].set_ylabel('[%s/H]' % elstr)
            ax[i].set_xlabel('$\mathbf{ T_{eff} }$')

        f  = mergeAxes(f)
        if save:
            plt.savefig('Thesis/pyplots/teff.ps')


    def feh(self,save=False,noratio=False):
        """
        Plot  [X/Fe] against [Fe/H]
        save    - saves the file
        noratio - Plot [X/H] against [Fe/H]
        """
        #pull in fitted abundances from tfit
        flags  = ['dn','dk']

        binmin,binwid,nbins = -0.5,0.1,11

        qtype= [('abund',float),('feh',float),('staterrlo',float),
                ('staterrhi',float),('pop_flag','|S10')]    
        
        bins = np.linspace(binmin,binmin+binwid*nbins,nbins)

        subplot = ((1,2))
        f = plt.figure( figsize=(6,6) )

        ax = []
        nel = self.params.nel
        lines = self.params.lines

        for i in range(nel):
            ax = appendAxes(ax,nel,i)

            p = getelnum.Getelnum(lines[i])           
            elstr = p.elstr
            cmd = '''
SELECT %s_abund,fe_abund,%s_staterrlo,%s_staterrhi,pop_flag 
FROM mystars
WHERE %s  AND pop_flag = "%s" 
''' % (elstr,elstr,elstr,postfit.globcut(elstr),'dn')
            self.cur.execute(cmd)
            arrthin = np.array(self.cur.fetchall(),dtype=qtype)
            
            arrthin['abund'] -= p.abnd_sol

            cmd = '''
SELECT %s_abund,fe_abund,%s_staterrlo,%s_staterrhi,pop_flag 
FROM mystars
WHERE  %s AND pop_flag = "%s" 
''' % (elstr,elstr,elstr,postfit.globcut(elstr),'dk')
            self.cur.execute(cmd)

            arrthick = np.array(self.cur.fetchall(),dtype=qtype)
            arrthick['abund'] -= p.abnd_sol
                                
            if noratio:
                ythin = arrthin['abund']
                ythick = arrthick['abund']
                ax[i].set_ylabel('[%s/H]' % (elstr) )
                savename = 'xonh'
            else:
                ythin = arrthin['abund'] -arrthin['feh']
                ythick = arrthick['abund'] -arrthick['feh']
                ax[i].set_ylabel('[%s/Fe]' % (elstr) )
                savename = 'xonfe'

            ### Compute Avg Thin disk in bins ###
            binind = np.digitize(arrthin['feh'],bins)
            binx,biny = [] , []
            for j in np.arange(nbins-1)+1:
                ind = (np.where(binind == j))[0]
                midbin = binmin+binwid*(j-0.5)
                binmean = np.mean(ythin[ind])
                nbin = len(ythin[ind]) # numer of points in a bin

                mederr = 0.5*(arrthin['staterrhi'][ind]-arrthin['staterrlo'][ind])
                pull = (ythin[ind] - binmean)/mederr
                binx.append(midbin)
                biny.append(binmean)

                print 'Bin %.2f: n = %i, stdpull =  %.2f ' % (midbin,nbin,np.std(pull))

                
            binx,biny = np.array(binx),np.array(biny)            
            ax[i].plot(arrthin['feh'],ythin,'bo')
            ax[i].plot(arrthick['feh'],ythick,'go')
            ax[i].plot(binx,biny,'rx-',lw=2,ms=5,mew=2)

            #plot typical errorbars
            yerr = np.array([s.std_dev() for s in self.stats[elstr]])
            if not noratio:
                yerr = np.sqrt(yerr**2 + p.feherr**2)
            yerr = yerr.reshape((2,1))

            ax[i] = errpt(ax[i],(0.8,0.8),xerr=p.feherr,yerr=yerr)
            ax[i].set_xlabel('[Fe/H]')                

        leg = ax[0].legend( ('Thin Disk','Thick Disk','Binned Thin Disk'), loc='best')

        f = mergeAxes(f)
        if save:
            plt.savefig('Thesis/pyplots/%s.ps' % savename)


    def abundhist(self,save=False,texcmd=False,uplim=False):
        """
        Plot the distributions of abundances.  Possibly not needed with the exo 
        plot.
        save - save the plot
        texcmd - returns what the tex command dumper wants

        """
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
                outex.append(r'$\text {%s}$& %i & %.2f & %.2f & %.2f & %.2f\\'
                             % (antxt,N,m,s,min,max))
            else:
                print 'N, mean, std, min, max' + antxt
                print '(%i,%f,%f,%f,%f)' % (N,m,s,min,max)
                

        f = mergeAxes(f)

        if texcmd:
            return nstars
        if save:
            plt.savefig('Thesis/pyplots/abundhist.ps')
            f = open('Thesis/tables/abundhist.tex','w')
            f.writelines(outex)

    def comp(self,save=False,texcmd=False):
        """
        Plots my results as a function of literature
        save - saves the file
        """
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
                           marker='o',ls='None',capsize=0,markersize=5)

            line = np.linspace(-3,3,10) # Plot 1:1 Correlation
            ax[i].plot(line,line)

            # Label and format plots
            ax[i].set_xlabel('[%s/H], %s' % (elstr,p.compref))
            ax[i].set_ylabel('[%s/H], This Work' % (elstr) )
            ax[i].set_xlim((-0.6,+0.6))
            ax[i].set_ylim((-0.6,+0.6))
            
        plt.draw()
        if texcmd:
            return texdict

        if save:
            plt.savefig('Thesis/pyplots/comp.ps')

    def exo(self,save=False,prob=True,texcmd=False):
        """
        Show a histogram of Carbon and Oxygen for planet harboring stars, and
        comparison stars.
        """
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

                ax[i].hist(x,bins=bins,weights=y)
                ax[i].errorbar(x,y,yerr=yerr,marker='o',elinewidth=2,ls='None')
            else:
                # Make histogram
                ax[i].hist([arrcomp,arrhost], bins=nbins,range=rng, histtype='bar',
                           label=['Comparison','Planet Hosts'])
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
                outex.append(r'$\text{[%s/H]}$ & %i & %.2f & %.2f & & %i & %.2f & %.2f & %s \\' % (elstr,nhost,mhost,shost,ncomp,mcomp,scomp,flt2tex.flt2tex(p,sigfig=1) ) )

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
            plt.savefig('Thesis/pyplots/exo.ps')
            f = open('Thesis/tables/exo.tex','w')
            f.writelines(outex)

        if texcmd:
            return statdict

    def cofe(self,save=False,texcmd=False):
        """
        Plots C/O as a function of Fe/H
        save - save the file
        """

        p = getelnum.Getelnum('O')
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
        ax.plot(arrcomp['feh'],c2ocomp,'bo')
        ax.plot(arrhost['feh'],c2ohost,'go')
        ax.legend(('Comparision','Hosts'),loc='best')
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('C/O')

        c2o = 10**(self.stats['C']-self.stats['O'])
        yerr = np.array([s.std_dev() for s in c2o]).reshape((2,1))

        ax = errpt(ax,(0.1,0.7),xerr=p.feherr,yerr=yerr)

        ax.set_ybound(0,ax.get_ylim()[1])
        ax.axhline(1.)
        plt.show()

        if save:
            plt.savefig('Thesis/pyplots/cofe.ps')
        return ax
        
    def compmany(self,elstr='o'):
        if elstr == 'o':
            tables = ['mystars','luck06','ramstars']
        if elstr =='c':
            tables = ['mystars','luck06','red06']

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
