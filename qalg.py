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


import itertools
import cPickle as pickle
import matplotlib.pylab as plt
from numpy import *
from numpy.random import random
import sys
from scipy.special import erfc

import ebls
import detrend
from keptoy import *

# Don't believe any faps that are below this value.  The extrapolation
# of the CDF is surely breaking down.

def init(tbase=[1000,1000] ,ntbase = 1,
         s2n = [10.,10.]   ,ns2n   = 1,
         ntr=20, null=False):
    """
    Initializes simulation parameters.

    Period is dithered by wper
    Phase is random.
    
    Random values are controlled by a seed value so that the run parameters
    are reproducable.
    """
    

    per = 200.
    wphase = 2*pi      # randomize phase over entire cycle
    wper   = per / 10. # dither period by 10%

    # Initialize test parameters    
    s2n   = log10( array(s2n) )
    tbase = log10( array(tbase))

    s2n = logspace(s2n[0],s2n[1],ns2n)
    tbase = logspace(tbase[0],tbase[1],ntbase)
    par = itertools.product(*[s2n,tbase])

    darr = []
    seed = 0

    for p in par:
        for i in range(ntr):            

            np.random.seed(seed)
            phase = wphase*random()
            dper   = wper*random()
            d = {'s2n':p[0],'tbase':p[1],'P':per+dper,'phase':phase,
                 'null':null,'seed':seed}

            darr.append(d) 
            seed += 1
        
    return darr


def genSynLC(darr):
    """
    Generate synthetic lightcurves with transits parametersspecified
    by darr
    """
    fl,tl = [],[]
    for d in darr:
        f,t = lightcurve(**d) 
    
        f = f.astype(float32)
        t = t.astype(float32)
        fl.append(f)
        tl.append(t)
        
    return tl,fl


def genEmpLC(darr,tdt,fdt):
    """
    Generate synthetic lightcurves with transits parameters specified
    by darr
    """
    fl,tl = [],[]
    for d in darr:
        f = inject(tdt,fdt,s2n=d['s2n'],P=d['P'],phase=d['phase']) 
    
        f = f.astype(float32)
        fl.append(f)
        tl.append(tdt)
        
    return tl,fl

def profile(tl,fl,PGrid,darr,save=None):
    """
    Will profile a transit search algorithm over a range of:
    - S/N
    - tbase
    
    Arguments
    func - a function the finds the period.  Must only accept f,t
    time series 
    darr - an array of dictionary arguments that will be passed to
    lightcurve`  These are the parameters for the fits. Must include:
    P     - period
    """
    res = eblspro(tl,fl,PGrid)


    if save != None:
        f = open(save,'wb')
        pickle.dump({'res':res,'darr':darr},f,protocol=2)
        
    return res

def dpack(PGrid,args):
    return {'PGrid':PGrid,'s2nGrid':s2nGrid}


def eblspro(tl,fl,PGrid,i):
    sys.stderr.write("%i" % i)
    s2nGrid = ebls.blsw(tl,fl,PGrid)
    out = {'PGrid':PGrid,'s2nGrid':s2nGrid}

    return out

def pprofile(tl,fl,PGrid,darr,save=None):

    """
    Multicore version of previous code
    """
    from IPython.parallel import Client
    rc = Client()
    lview = rc.load_balanced_view()
    nsim = len(tl)
    PGridList = [PGrid for i in range(nsim)]
    counter = range(nsim)

    res = lview.map(eblspro,tl,fl,PGridList,counter,block=True)

    if save != None:
        f = open(save,'wb')
        pickle.dump(
            {'res':res,'darr':darr},
            f,protocol=2)
        f.close()

    return res


def loadrun(path):
    """
    Load the values from a Pickle file
    """
    out = pickle.load(open(path,'r'))
    return out['darr'],out['res']


def s2n2fap(PGrid,s2nGrid):
    """
    """
    tdurGrid = a2tdur(P2a(PGrid))

    return log10(0.5*(erfc(s2nGrid/sqrt(2)) )*(PGrid/tdurGrid*5))
    




fapfloor = 0.01;
def peakfap(res):
    """    
    nper - period array for null run
    npow - power array for null run
    per  - period array for lightcurve spectrum
    pow  - power array for lightcurve spectrum
    """

    

    return miper,mifap

def bper(PGrid,FAPGrid):
    """
    Return the best period and associated FAP
    """


    idMi = FAPGrid.argmin()
    bp = PGrid[idMi]
    fap = FAPGrid[idMi]

    return bp,fap

def addFAPGrid(res):
    for r in res:
        r['FAPGrid'] = s2n2fap(r['PGrid'],r['s2nGrid']) 
    return res
                     
def addbper(res):
    for r in res:
        r['bP'],r['bFAP'] = bper(r['PGrid'],r['FAPGrid']) 
    return res


def fap_s2n(darr,res,failthresh=0.1):
    """
    Show how the FAP changes as a function of s2n    
    
    in order to claim that we've found the right period, it must be
    within failthresh of the input value.

    """
    fig = plt.gcf() 

    s2n = [ d['s2n'] for d in darr ]
    us2n = unique( s2n  )
    ns2n = len(us2n)

    iP = array([d['P'] for d in darr])

    oP  = array( [r['bP'] for r in res  ] )
    fap = array( [r['bFAP'] for r in res  ] )
    
    for i in range(ns2n):
        s = us2n[i]
        bools2n  = (s2n == s)
        boolfail = abs(oP-iP)/iP > failthresh


        x = fap

        ax = fig.add_subplot(ns2n,1,i+1)
        bins = linspace(-40,0,21)
        
        ax.hist( x[where(~boolfail & bools2n)] ,color='g',bins=bins,
                 label='Good')
        ax.hist( x[where(boolfail & bools2n)] ,color='r',alpha=0.8,
                 bins=bins,label='Fail')

        plt.legend(title='S/N - %0.2f' %  s , loc='best')

    ax.set_xlabel('log(FAP)')
    plt.show()
 

def iPoP(darr,res):
    """
    Plot input period versus output period.
    """

    fig = plt.gcf() 

    s2n = [ d['s2n'] for d in darr ]
    us2n = unique( s2n  )
    ns2n = len(us2n)

    iP = array([r['P'] for r in darr])

    oP  = array( [r['bP'] for r in res  ] )
    fap = array( [r['bFAP'] for r in res  ] )
    



    for i in range(ns2n):
        s = us2n[i]
        bools2n  = (s2n == s)

        ax = fig.add_subplot(ns2n,1,i+1)

        ax.plot( iP[ where(bools2n) ]  , oP[ where(bools2n) ], '.b',ms=3)
        ax.set_xlabel('Input Period')
        ax.set_ylabel('Best Period')

        tt = "S/N  - %.2f" % s


        left = ax.get_xlim()[0]
        top = ax.get_ylim()[1]

        ax.text(left,top,tt,bbox=dict(facecolor='white'),fontsize='large',
                va='top',ha='left')

    plt.show()