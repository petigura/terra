"""
Motivation:

This module contains functions for determining the `efficacy' of
transit detection algorithms.  

Goals:
- Manage the creation of monte carlo data.
- Profiling different algorithms in a systematic way.
- Storing the results.

Requirements:
- Functions should should take only t,f as required arguments

What is Efficacy?
-----------------
- The metric for the efficacy of a function will be the number of
correctly identified transit signatures over the number of mock data
sets.
- Function must identify the correct trial period to within 1%
- Function must determine the phase of the transit to 1%
"""

import numpy as np
from numpy import pi
from numpy.random import random
from itertools import product

from keptoy import lightcurve


def init(tbase=[30,600]   ,ntbase = 30,
         s2n = [100.,100.], ns2n = 1,
         ntr=20):


    per = 10.
    wphase = 2*pi      # randomize phase over entire cycle
    wper   = per / 10. # dither period by 10%

    # Initialize test parameters    
    s2n   = np.log10( np.array(s2n) )
    tbase = np.log10( np.array(tbase))

    s2n = np.logspace(s2n[0],s2n[1],ns2n)
    tbase = np.logspace(tbase[0],tbase[1],ntbase)

    par = product(*[s2n,tbase])

    darr = []

    dtemp  = { 's2n':0.,'tbase':0.,'P':0.,'phase':0.}
    for p in par:
        for i in range(ntr):
            phase = wphase*random()
            dper   = wper*random()
            d = { 's2n':p[0],'tbase':p[1],'P':per+dper,'phase':phase}
            darr.append(d) 
        
    return darr

def profile(darr,func):
    """
    Will profile a transit search algorithm over a range of:
    - S/N
    - tbase
    
    Arguments
    func - a function the finds the period.  Must only accept f,t
    time series 
    darr - an array of dictionary arguments that will be passed to
    `lightcurve`  These are the parameters for the fits. Must include:
     P     - period
     phase - phase
     
     """

    print """
P    bper dP     iph  oph  dph    pass
---- ---- ------ ---- ---- ------ ----"""

    parr,farr,ol,flux,time = [],[],[],[],[]
    
    nruns = len(darr)
    count = 0 
    for d in darr:
        # generate the lightcurve with the specified parameters.
        count += 1 
        if np.mod(count*100,nruns) == 0:
            print """
%i %%
""" % int(100.*count/nruns)

        
        o = d.copy()
        f,t = lightcurve(**d) 
        try:
            res = func(t,f)
            bper = res['bper'] 
            iph  = d['phase']
            oph  = res['phase']

            o['bper']  = bper
            o['phase'] = oph
            
            # Is it a detection
            if np.abs( o['bper'] - d['P']) >  d['P'] / 100.:                
                o['pass'] = False
            elif np.abs( o['phase'] - d['phase']) >  np.pi/ 10.:
                o['pass'] = False
            else:
                o['pass'] = True

                print '%.1f %.1f %+.0e %.2f %.2f %+.0e %s' % (
                    d['P'], bper, d['P'] - bper,      
                    d['phase'],oph, d['phase'] - oph,
                    o['pass']              
                    )

        except ValueError:
            print "Value Error"
            o['bper']  = bper = None
            o['phase'] = oph  = None
            o['pass']  = False


        parr.append(res['p'])
        farr.append(res['farr'])
        flux.append(f)
        time.append(t)


        ol.append(o)    

    # Convert the list of dicts in to a record array

    tlist = [tuple([val for key,val in o.items()]) for o in ol]
    names = [key for key,val in ol[0].items()]
    form = [type(val) for key,val in ol[0].items()]
    ol = np.array( tlist , {'formats':form,'names':names} )

    # matrix out of all the power spectra
    return ol,parr,farr,flux,time


def h1d(ol,key):
    """
    accepts a record array of the profiler results
    """
    
    # value to plot.  There will be multiple runs with the same
    # configuration so we only want the uniqu elements.
    x = ol[key]
    x = np.unique(x)
    eta = np.zeros( len(x) ) 
    n = np.zeros( len(x) ) 
    for i in range( len(x) ) :
        cidx = (np.where(ol[key] == x[i] ) )[0]
        temp = ol[cidx]
        ntr   = len(temp)
        temp = temp[( np.where(temp['pass']) )[0]]
        npass = len(temp)
        eta[i] = 1.*npass/ntr

    return x,eta


def h2d(xkey,ykey):
    pass


def wrap():

    import pgram
    d = init()
    ol = profile(d,pgram.blswrap)
    return h1d(ol,'tbase')
