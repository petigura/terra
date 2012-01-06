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
import atpy
import copy

import ebls
import detrend
from keptoy import *

def init(**kwargs):
    """
    Initializes simulation parameters.

    Period is dithered by wper
    Phase is random.
    
    *kwargs - key word arguments to pass to inject.

    Usage
    -----
    init(P=200,s2n=([10,20],5),tbase=500).  

    - List of dictionaries specifying 5 runs centered around P = 200,
      5 values of s2n ranging from 10 to 20, and tbase=500.
    

    """
    keys=kwargs.keys()
    assert keys.count('P') == 1, "Must specify period"

    P = kwargs['P']
    wP   = P / 10.   # dither period by 10%
    arrL = []

    keys.remove('P')

    for k in keys:
        arg = kwargs[k]
        if type(arg) is not tuple:
            arg = (arg,1)

        n     = arg[1]
        if type(arg[0]) is list:
            range = arg[0]
            arr   = logspace( log10(range[0]) , log10(range[1]) , n)
        else:
            arr    = empty(n)
            arr[:] = arg[0]

        arrL.append(arr)

    par = itertools.product(*arrL)
    darr = []
    seed = 0
    for p in par:        
        np.random.seed(seed)
        epoch = P*random()
        dP   = wP*random()
        d = {'P':P+dP,'epoch':epoch,'seed':seed}
        for k,v in zip(keys,p):
            d[k] = v
        
        darr.append(d) 
        seed += 1

    return darr

def genSynLC(darr):
    """
    Generate Synthetic Lightcurves

    Parameters
    ----------
    darr : List of dictionaries specifying the LC parameters.

    Returns
    -------
    tl   : List of time arrays
    fl   : List of flux arrays
    
    """
    fl,tl = [],[]
    for d in darr:
        f,t = lightcurve(**d) 
    
        f = f.astype(float32)
        t = t.astype(float32)
        fl.append(f)
        tl.append(t)
        
    return tl,fl

def tab2dl(t):
    """
    Convert a table to a list of dictionaries
    """
    dl = []
    columns = t.columns.keys
    nrows = len(t.data[ columns[0] ] )
    for i in range( nrows ):
        d = {}
        for k in columns:
            d[k] = t[k][i]

        dl.append(d)
    return dl

def dl2tab(dl):
    """
    Convert a list of dictionaries to an atpy table.
    """
    t = atpy.Table()
    keys = dl[0].keys()
    for k in keys:
        data = array( [ d[k] for d in dl ] )
        t.add_column(k,data)

    return t

def genEmpLC(darr0,tdt,fdt):
    """
    Generate Synthetic Lightcurves

    Parameters
    ----------
    darr : List of dictionaries specifying the LC parameters.
    tdt  :
    fdt  :


    Returns
    -------
    tl   : List of time arrays
    fl   : List of flux arrays
    
    """

    darr = copy.deepcopy(darr0)

    fl,tl = [],[]
    for d in darr:
        d.pop('seed')
        d.pop('tbase')

        f = inject(tdt,fdt,**d)
        f = f.astype(float32)
        fl.append(f)
        tl.append(tdt)
        
    return tl,fl

def profile(tl,fl,PGrid,func,par=False):
    """
    Will profile a transit search algorithm over a range of:
    - S/N
    - tbase
    
    Parameters
    ----------
    tl : List of time measurements
    fl : List of flux measurements
    PGrid : List trial periods

    func : Function that is being profiled. Must have the following
           signature:
           func(t,f,PGrid,counter)

    """
    nsim = len(tl)
    PGridList = [PGrid for i in range(nsim)]
    counter = range(nsim)

    if par:
        from IPython.parallel import Client
        rc = Client()
        lview = rc.load_balanced_view()
        res = lview.map(func,tl,fl,PGridList,counter,block=True)
    else:
        res = map(func,tl,fl,PGridList,counter)        

    return res

def saverun(darr,res,path):
    f = open(path,'wb')
    pickle.dump({'res':res,'darr':darr},f,protocol=2)
    f.close()


def s2n2fap(PGrid,s2nGrid):
    """
    """
    tdurGrid = a2tdur(P2a(PGrid))
    return log10(0.5*(erfc(s2nGrid/sqrt(2)) )*(PGrid/tdurGrid*5))

fapfloor = 0.01;

# If best fit period is off by more than this in a fractional sense,
# label it a failure.

failthresh = 0.01 
def PP(tset):
    """
    Post processing

    Change so we're dealing with tables.
    """
    tres = tset.RES
    tpar = tset.PAR

    nrows = len(tres)

    idMa = tset.RES.s2nGrid.argmax(axis=1)

    try:
        tres.add_empty_column('bP',np.float)
        tres.add_empty_column('bph',np.float)
        tres.add_empty_column('fail',np.bool)
    except:
        pass

    for i in range(nrows):
        id = idMa[i]
        tres.bP[i] = tres.PGrid[i][id]
        tres.bph[i] = tres.phGrid[i][id]
        tres.fail[i] = abs(tpar.P[i] - tres.bP[i])/tpar.P[i] > failthresh

    return tset

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
 

def iPoP(iP,oP):
    """
    Plot input period versus output period.
    """

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


def loadrun(path):
    """
    Load the values from a Pickle file
    """
    out = pickle.load(open(path,'r'))
    return out['darr'],out['res']

