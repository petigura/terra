import getelnum
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os

def globcut(elstr,table='mystars'):    
    p = getelnum.Getelnum(elstr)
    vsinicut = p.vsinicut
    teffrng = p.teffrng

    if (elstr == np.array(['O','C'])).any() is False:
        return ''
    
    locut = -0.2
    hicut = 0.2
    cut = """
tab.vsini < %d AND tab.%s_abund > 0  AND 
tab.%s_scatterlo > %.2f AND tab.%s_scatterhi < %.2f AND 
tab.teff > %d AND tab.teff < %d""" % (vsinicut,elstr,elstr,locut,elstr,hicut,teffrng[0],teffrng[1])
   
    cut = cut.replace('tab',table)
    return cut

def tfit(line):
    """
    Correct for temperature systematics.  Fit a polynomial to (teff,abund)
    and require that the corrected solar value be 0.  We cut on vsini, 

    returns:
    (fitabund,fitpar,t,abund)
    fitabund - the temperature corrected abundance
    fitpar   - the parameters to the polynomial fit
    t        - the temperature array
    abund    - the non-temp-corrected abundances

    """
    deg  = 3 # fit with a 3rd degree polynomial
    #define abundance for the particular line we're looking at
    p = getelnum.Getelnum(line)
    elstr = p.elstr

    conn = sqlite3.connect(os.environ['STARSDB'])
    cur = conn.cursor()

    #pull in the abundances and the non-corrected abundances
    cmd = 'SELECT '+elstr+'_abund_nt,teff FROM mystars WHERE '+globcut(elstr)
    cur.execute(cmd)
    arr = np.array(cur.fetchall() ) 
    abund,t = arr[:,0],arr[:,1]
    abund = abund - p.abnd_sol

    #fit the points
    fitpar = np.polyfit(t,abund,deg)
    #subtract out the fit, while requiring that the solar value be 0.
    fitpar[deg] = fitpar[deg] - np.polyval(fitpar,p.teff_sol)
    fitabund = abund - np.polyval(fitpar,t)
    return (fitabund,fitpar,t,abund)

def applytfit():
    lines = [6300,6587]
    conn = sqlite3.connect(os.environ['STARSDB'])
    cur = conn.cursor()


    for line in lines:
        #pull the data
        

        p = getelnum.Getelnum(line)
        elstr = p.elstr

        cut = ' WHERE '+elstr+'_abund > 0 ' #do not apply correction to the -99s
        cmd = ' SELECT id FROM mystars '+cut
        
        cur.execute(cmd)
        ids = (np.array( cur.fetchall() )).flatten()

        cmd = ' SELECT teff,'+elstr+'_abund FROM  mystars '+cut
        cur.execute(cmd)
        arr = np.array( cur.fetchall() )
        t,abund0 = arr[:,0],arr[:,1]

        x,fitpar,x,x = tfit(line)

        abund = abund0 - np.polyval(fitpar,t)
        
        for i in range(len(ids)):
            cmd = 'UPDATE mystars SET %s_abund = %f WHERE mystars.id = %s' %\
                (elstr,abund[i],ids[i])
            if i in [0,1,2,3]:
                print cmd+' '+str(abund0[i])

            cur.execute(cmd)

    conn.commit() #commit the changes
    cur.close() #commit the changes

def plotuplim(stars):
    """
    see where the upperlimits live
    """
    lines = [6300,6587]
    i = 1
    plt.clf()

    f = plt.figure()
    for line in lines:
        f.add_subplot(1,2,i)

        #cut off bad fits
        p = getelnum.Getelnum(line)
        fitpass = fitbool(stars,line)
        vpass = vbool(stars,line)
        ul = ulbool(stars,line)

        goodidx  = np.where(~ul & fitpass & vpass)
        ulidx = np.where(ul & fitpass & vpass)

        exec('abund  = stars.'+p.abundfield)    

        plt.scatter(stars.teff[goodidx],abund[goodidx])
        plt.scatter(stars.teff[ulidx],abund[ulidx],color='red')

        i+=1

def plottfit(stars):

    """
    A quick look at the fits to the temperature
    """

    line = [6300,6587]
    subplot = ((1,3),(2,4))
    plt.clf()
    f = plt.figure()
    for i in range(2):
        o = getelnum.Getelnum(line[i])           
        fitabund, fitpar, t,abund = tfit(stars,line[i])    
        tarr = np.linspace(t.min(),t.max(),100)
        
        f.add_subplot(2,2,subplot[i][0])
        plt.scatter(t,abund)
        plt.scatter(o.teff_sol,0.,color='red')
        plt.plot(tarr,np.polyval(fitpar,tarr),lw=2,color='red')
        
        f.add_subplot(2,2,subplot[i][1])
        plt.scatter(t,fitabund)
        plt.scatter(o.teff_sol,np.polyval(fitpar,o.teff_sol),color='red')


def names(el):
    """
    return a bunch of names
    """
    conn = sqlite3.connect(os.environ['STARSDB'])
    cur = conn.cursor()

    #pull in the abundances and the non-corrected abundances
    cmd = 'SELECT name FROM mystars WHERE '+globcut(el)
    cur.execute(cmd)
    arr = np.array(cur.fetchall(),dtype='|S10') 
    list = ''
    for a in arr:
        list = list+a[0]+' '

    return list
