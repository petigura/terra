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
    
    **kwargs - key word arguments to pass to inject.
    Choose from:
    - df or s2n
    - epoch or phase
    - P

    Usage
    -----
    init(P=200,s2n=([10,20],5),tbase=500).  

    - List of dictionaries specifying 5 runs centered around P = 200,
      5 values of s2n ranging from 10 to 20, and tbase=500.
    """
    keys=kwargs.keys()
    assert keys.count('P') == 1, "Must specify period"

    P = kwargs['P']
    wP   =  0.1   # dither period by 10%
    arrL = []

    mult = kwargs['n']
    keys.remove('n')

    for k in keys:
        arg = kwargs[k]
        if type(arg) is not tuple:
            arg = (arg,1)

        n     = arg[1]
        if type(arg[0]) is list:
            rng = arg[0]
            arr   = logspace( log10(rng[0]) , log10(rng[1]) , n)
        else:
            arr    = empty(n)
            arr[:] = arg[0]

        arrL.append(arr)

    par = itertools.product(*arrL)
    darr = []
    seed = 0
    for p in par:        
        for i in range(mult):
            d = {}
            for k,v in zip(keys,p):
                d[k] = v

            d['Pblock'] = d['P']
            np.random.seed(seed)
            d['P'] = d['P']*(1 + wP*random() ) 
            d['epoch'] = d['P']*random()
            d['seed'] = seed
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


def ROC(t):
    """
    Construct a reciever operating curve for a given table.

    Table must be homogenous (same star, same input parameters).
    """
    
#    fomL = logspace( log10(t.os2n.min()),log10( t.os2n.max()), 1000   )
    fomL = unique(t.os2n)
    print t.os2n.min(), t.os2n.max(),
    etaL = [] 
    fapL = []
    n  = t.data.size
    for fom in fomL:
        eta = float((t.where( (t.os2n > fom) & t.bg )).data.size) / n
        fap = float((t.where( (t.os2n > fom) & ~t.bg )).data.size) / n

        etaL.append(eta)
        fapL.append(fap)
               
    return fapL,etaL
