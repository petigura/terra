import readstars,postfit,matchstars,starsdb,getelnum,flt2tex
import scipy
import scipy.stats as stats
from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pdb
import uncertainties
from uncertainties import unumpy
import os

class Plotgen():

    def __init__(self):
    ###Pull up database.
        self.conn = sqlite3.connect(os.environ['STARSDB'])
        self.cur = self.conn.cursor()    
        elements = ['O','C']    
        self.stats = {}

        for elstr in elements:
            cmd = 'SELECT %s_abund,%s_staterrlo,%s_staterrhi FROM mystars '\
                % (elstr,elstr,elstr)
            wcmd = ' WHERE '+postfit.globcut(elstr)
            self.cur.execute(cmd+wcmd)

            temptype =[('abund',float),('staterrlo',float),('staterrhi',float)]
            arr = np.array(self.cur.fetchall(),dtype=temptype)
            m,slo,shi = np.mean(arr['abund']),np.abs(np.median(arr['staterrlo'])),np.median(arr['staterrhi'])                
            self.stats[elstr] = unumpy.uarray(([m,m],[slo,shi])) 


    def errpt(self,ax,coord,xerr=None,yerr=None):
        """
        Plot a point that shows an error 
        ax - axis object to manipulate and return
        coord - the coordinates of error point (in device coordinates)
        """
        inv = ax.transData.inverted()
        pt = inv.transform( ax.transAxes.transform( coord ) )
        ax.errorbar(pt[0],pt[1],xerr=xerr,yerr=yerr,
                    capsize=0,
                    color='red',
                    elinewidth = 2)
        return ax
   
    def tfit(self,save=False,fitres=False):
        """
        A quick look at the fits to the temperature
        save - saves the plot
        fitres - plots the residuals
        """
        line = [6300,6587]
        nel = len(line)
        f = plt.figure( figsize=(6,6) )
        f.subplots_adjust(hspace=0.0001)

        ax = []
        for i in range(nel):
            p = getelnum.Getelnum(line[i])           
            elstr = p.elstr.upper()

            if i is 0:
                ax.append(plt.subplot(nel,1,i+1))
            else:
                ax.append(plt.subplot(nel,1,i+1,sharex=ax[0]))
                
            fitabund, fitpar, t,abund = postfit.tfit(line[i])    
            if fitres:
                abund = fitabund

            tarr = np.linspace(t.min(),t.max(),100)        
            ax[i].set_ylabel('[%s/H]' % elstr)
            ax[i].scatter(t,abund,color='black',s=10)
            ax[i].scatter(p.teff_sol,0.,color='red',s=30)
            ax[i].plot(tarr,np.polyval(fitpar,tarr),lw=2,color='red')        

            yerr = np.array([s.std_dev() for s in self.stats[elstr]])
            yerr = yerr.reshape((2,1))

            ax[i] = self.errpt(ax[i],(0.9,0.9),xerr=p.tefferr,yerr=yerr)
            ax[i].set_ylabel('['+elstr+'/H]')

            if i is nel-1:
                ax[i].set_xlabel('$\mathbf{ T_{eff} }$')
            else:
                ax[i].set_xticklabels('',visible=False)
                #remove overlapping zeroes
                yticks = ax[i].get_yticks()[1:]
                ax[i].set_yticks(yticks)

        if save:
            plt.savefig('Thesis/pyplots/teff.ps')


    def feh(self,save=False,noratio=False):
        """
        Show the trends of X/Fe as a function of Fe/H.
        save - saves the file
        noratio - plots X/H as opposed to X/Fe
        """
        #pull in fitted abundances from tfit

        lines = [6300,6587]
        nel = len(lines)
        flags  = ['dn','dk']

        binmin = -0.5
        binwid = 0.1
        nbins = 11
        qtype= [('abund',float),('feh',float),('staterrlo',float),
                ('staterrhi',float),('pop_flag','|S10')]    
        
        bins = np.linspace(binmin,binmin+binwid*nbins,nbins)

        subplot = ((1,2))
        f = plt.figure( figsize=(6,6) )

        f.subplots_adjust(hspace=0.0001)
        ax = []


        for i in range(nel):
            if i is 0:
                ax.append(plt.subplot(nel,1,i+1))
            else:
                ax.append(plt.subplot(nel,1,i+1,sharex=ax[0]))


            p = getelnum.Getelnum(lines[i])           
            elstr = p.elstr
            cmd = 'SELECT %s_abund,fe_abund,%s_staterrlo,%s_staterrhi,pop_flag FROM mystars ' % (elstr,elstr,elstr)
            wcmd = ' WHERE '+postfit.globcut(elstr)+' AND pop_flag = "dn"'
            self.cur.execute(cmd+wcmd)
            arrthin = np.array(self.cur.fetchall(),dtype=qtype)
            
            arrthin['abund'] -= p.abnd_sol

            wcmd = ' WHERE '+postfit.globcut(elstr)+' AND pop_flag = "dk"'
            self.cur.execute(cmd+wcmd)

            arrthick = np.array(self.cur.fetchall(),
                           dtype=qtype)
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


            ax[i] = self.errpt(ax[i],(0.9,0.9),xerr=p.feherr,yerr=yerr)

            if i is nel-1:
                ax[i].set_xlabel('[Fe/H]')                
            else:
                ax[i].set_xticklabels('',visible=False)
                #remove overlapping zeroes
                yticks = ax[i].get_yticks()[2:] #hack
                ax[i].set_yticks(yticks)
            
        leg = ax[0].legend( ('Thin Disk','Thick Disk','Binned Thin Disk'), loc='best')
        if save:
            plt.savefig('Thesis/pyplots/%s.ps' % savename)


    def abundhist(self,save=False,texcmd=False,uplim=False):
        """
        Plot the distributions of abundances.  Possibly not needed with the exo 
        plot.
        save - save the plot
        texcmd - returns what the tex command dumper wants

        """
        #pull in fitted abundances from tfit

        line = [6300,6587]

        subplot = ((1,2))
        f = plt.figure( figsize=(6,6) )

        f.subplots_adjust(hspace=0.0001)
        ax1 = plt.subplot(211)
        ax1.set_xticklabels('',visible=False)
        ax1.set_yticks(np.arange(0,200,50))

        ax2 = plt.subplot(212,sharex=ax1)
        ax2.set_yticks(np.arange(0,200,50))
        ax2.set_xlabel('[X/H]')
        ax = (ax1,ax2)
        nstars  = []
        outex = []

        for i in range(2):
            p = getelnum.Getelnum(line[i])           
            elstr = p.elstr
            
            cut = postfit.globcut(elstr,uplim=uplim)
            cmd = 'SELECT %s_abund from MYSTARS WHERE %s'% (elstr,cut)
            self.cur.execute(cmd)


            out = self.cur.fetchall()
            abund = np.array(out,dtype=float).flatten()-p.abnd_sol

            ax[i].set_ylabel('Number of Stars')
            ax[i].hist(abund,range=(-1,1),bins=20,fc='gray')
            ax[i].set_ylim(0,200)

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

        tables = [['ben05'],['luckstars']]
        offset = [[0],[8.5]]
        literr = [[0.06],[0.1]]

        lines = [6300,6587]
        color = ['blue','red','green']


        f = plt.figure( figsize=(6,8) )
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        ax1.set_xlabel('[O/H], Bensby 2005')
        ax2.set_xlabel('[C/H], Luck 2006')

        ax1.set_ylabel('[O/H], This Work')
        ax2.set_ylabel('[C/H], This Work')
        ax = [ax1,ax2]
        ncomp = []
        stdcomp = []

        for i in [0,1]:        
            xtot =[] #total array of comparison studies
            ytot =[] #total array of comparison studies

            p = getelnum.Getelnum(lines[i])
            elstr = p.elstr
            abnd_sol = p.abnd_sol

            print abnd_sol
            for j in range(len(tables[i])):
                table = tables[i][j]            

                #SELECT
                cmd = """
SELECT DISTINCT 
mystars.%s_abund,mystars.%s_staterrlo,
mystars.%s_staterrhi,%s.%s_abund """ % (elstr,elstr,elstr,table,elstr)
                if table is 'luckstars':
                    cmd += ','+table+'.c_staterr '

                #FROM WHERE
                cmd +=""" 
FROM mystars,%s WHERE mystars.oid = %s.oid 
AND %s.%s_abund IS NOT NULL AND %s"""  % (table,table,table,elstr,postfit.globcut(elstr))
                if table is 'luckstars':
                    cmd = cmd+' AND '+table+'.c_staterr < 0.3'

                self.cur.execute(cmd)
                arr = np.array(self.cur.fetchall())
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
                n = len(x)
                ncomp.append(n)
                print str(n) + 'comparisons'

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

            S = np.std(xtot[0]-ytot[0])
            print S
            stdcomp.append(S)

        plt.draw()
        if texcmd:
            return ncomp,stdcomp

        if save:
            plt.savefig('Thesis/pyplots/comp.ps')


    def exo(self,save=False,prob=True,texcmd=False):
        """
        Show a histogram of Carbon and Oxygen for planet harboring stars, and
        comparison stars.
        """
        f = plt.figure( figsize=(6,8) )

        f.subplots_adjust(hspace=0.0001)
        ax1 = plt.subplot(211)

        elements = ['O','C','Fe']
        nel = len(elements)
        sol_abnd = [8.7,8.5,0]

        ax = []  #empty list to store axes
        outex = []
        statdict = { 'host':{},'comp':{} }
        for i in range(nel): #loop over the different elements
            if i is 0:
                ax.append(plt.subplot(nel,1,i+1))
            else:
                ax.append(plt.subplot(nel,1,i+1,sharex=ax[0]))

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

            if i is nel-1:
                ax[i].set_xlabel('[X/H]')
                ax[i].set_ylabel('% Stars with Planets')
            else:
                ax[i].set_xlabel('['+elstr+'/H]')
                ax[i].set_xticklabels('',visible=False)
                #remove overlapping zeroes
                yticks = ax[i].get_yticks()[1:]
                ax[i].set_yticks(yticks)


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
                print elstr+' in stars w/  planets: N = %i Mean = %f Std %f ' \
                    % (nhost,mhost,shost)
                print elstr+' in stars w/o planets: N = %i Mean = %f Std %f ' \
                    % (ncomp,mcomp,scomp)
                print 'KS Test: D = %f  p = %f ' % (D,p)


                statdict['host'][elstr] = {'n':nhost,'m':mhost,'s':shost}
                statdict['comp'][elstr] = {'n':ncomp,'m':mcomp,'s':scomp}


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

        ax = self.errpt(ax,(0.1,0.7),xerr=p.feherr,yerr=yerr)

        ax.set_ybound(0,ax.get_ylim()[1])
        ax.axhline(1.)
        plt.show()

        if save:
            plt.savefig('Thesis/pyplots/cofe.ps')
        return ax
        
    def compmany(self,elstr='o'):
        if elstr == 'o':
            tables = ['mystars','luckstars','ramstars']
        if elstr =='c':
            tables = ['mystars','luckstars','red06']

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




