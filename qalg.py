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
import tfind
import keptoy
from scipy import ndimage as nd

from config import *

import sqlite3
import pandas

def mc_init():
    """
    Return a list of simulation parameters.

    """
    n = 100

    # Load in the star list
    con = sqlite3.connect('eb10k_clean.db')
    cur = con.cursor()
    cmd = "SELECT skic,a1,a2,a3,a4 from b10k_clean"
    cur.execute(cmd)
    res = cur.fetchall()
    res = pandas.DataFrame(res,columns=['skic','a1','a2','a3','a4'])


    # Randomly draw n stars from the list
    stars = res.ix[np.random.random_integers(0,res.shape[0],n)]

    Plo,Phi = 5,50    
    P = random(n)
    P = (log10(Phi)-log10(Plo)) * P + log10(Plo)
    P = 10**P
    stars['P'] = P

    dflo,dfhi = 50,500
    df = random(n)
    df = (log10(dfhi)-log10(dflo)) * df + log10(dflo)
    df = 10**df
    stars['df'] = df

    stars['b']     = random(n)
    stars['phase'] = random(n) 
    stars.index    = np.arange(n)
    return stars

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
    init(P=200,df=([100,400],5),tbase=500,KIC=[8144222,8409753]).  

    - List of dictionaries specifying 5 runs centered around P = 200,
      5 values of s2n ranging from 10 to 20, and tbase=500.
    """
    keys=kwargs.keys()
    assert keys.count('P') == 1, "Must specify period"
    assert keys.count('t0') == 1, "Specify t0 kwarg."
    assert keys.count('KIC') == 1, "Specify KIC number."
    assert kwargs['df'] > 1, "df should be in units of ppm"

    P = kwargs['P']
    t0 = kwargs['t0']
    wP   =  0.1   # dither period by 10%
    arrL = []

    if keys.count('n') == 1:
        mult = kwargs['n']
        keys.remove('n')
    else:
        mult = 1

    # Convert KW to list
    for k in keys:
        arg = kwargs[k]
        if type(arg) is tuple:
            rng = arg[0]
            arr   = logspace( log10(rng[0]) , log10(rng[1]) , arg[1])
        elif type(arg) is list:
            arr = arg
        else:
            arr = [arg]
        arrL.append(arr)

    par = itertools.product(*arrL)
    darr = []
    seed = 0
    for p in par:        
        for i in range(mult):
            d = {}
            for k,v in zip(keys,p):
                d[k] = v

            d['Pblock'] = int(d['P'])
            d['df'] = int(d['df'])

            np.random.seed(seed)
            d['P'] = d['P']*(1 + wP*random() ) 
            d['epoch'] = t0 + d['P']*random()
            d['tdur'] = a2tdur(P2a(d['P']))
            d['seed'] = seed
            darr.append(d) 

            seed += 1

    return darr

### Quick converters for various formats ###

def tab2dl(t):
    """
    Convert a table to a list of dictionaries
    """
    rec = tab.data
    dl  = [rec2d(r) for r in rec]
    return dl

def dl2tab(dL):
    """
    Convert a list of dictionaries to an atpy table.
    """
    rec = [d2rec(d) for d in dL]
    rec = np.hstack(rec)
    t   = rec2tab(rec)
    return t

def rec2tab(rec):
    """
    Convert a record array into an atpy table.
    """
    t = atpy.Table()
    keys = rec.dtype.names
    for k in keys:
        t.add_column(k,rec[k])
    return t

def rec2d(rec):
    """
    Convert a record into a dictionary.
    """

    keys = rec.dtype.names
    d = {}
    for k in keys:
        d[k] = rec[k]
    return d

def d2rec(d):
    """
    Convert a dictionary into a length 1 record array.
    """
    keys = d.keys()
    typeL = [type(v) for v in  d.values()]
    dtype = zip(keys,typeL)
    return np.array( tuple(d.values()) ,dtype=dtype )
    

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
               
    return fapL,etaL,fomL

def bg(P,oP,epoch,oepoch):
    """
    Did we correctly identify period
    """
    return ( abs(P - oP)/P < Plim ) & ( abs(epoch - oepoch) < epochlim )

def bharm(P,oP):
    Pharm = P * np.array([0.5,2])
    return (abs( oP / Pharm - 1) < Plim).any()

def bwin(tLC,P,epoch):
    """
    Boolean Window
    
    Did a particular combination of (P,epoch) get excluded based on the window?

    Parameters
    ----------

    """
    # Discard cadences that are too high.
    dM     = ma.masked_array(tLC.dM6,mask=tLC.dM6mask)

    Pcad   = round(P/keptoy.lc)
    res    = tfind.ep(tLC.t[0],dM,Pcad)
    winG   = res['win']

    # Find the first epoch after time = 0 .
    epochG = np.remainder(res['epoch'],P)
    idnn   = np.argmin( abs(epochG - epoch) )
    bwin   = winG[idnn].astype(bool)
    return bwin


