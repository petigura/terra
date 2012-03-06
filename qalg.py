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

Plim =  0.001   # Periods must agree to this fractional amount
epochlim =  0.1 # Epochs must agree to 0.1 days    

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

def rec2tab(rec):
    """
    Convert a list of dictionaries to an atpy table.
    """
    t = atpy.Table()
    keys = rec.dtype.names
    for k in keys:
        t.add_column(k,rec[k])
    return t

def rec2d(rec):
    keys = rec.dtype.names
    d = {}
    for k in keys:
        d[k] = rec[k]
    return d


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

def bg(P,oP,epoch,oepoch):
    """
    Did we correctly identify period
    """
    return ( abs(P - oP)/P < Plim ) & ( abs(epoch - oepoch) < epochlim )

def alias(P,oP):
    Palias = P * np.array([0.5,2])
    return (abs( oP / Palias - 1) < Plim).any()

def window(tLC,P,epoch):
    f = tLC.f

    bK,boxK,tK,aK,dK = tfind.GenK( 20 )
    dM = nd.convolve1d(f,dK)

    # Discard cadences that are too high.
    dM = ma.masked_outside(dM,-1e-3,1e-3)
    f = ma.masked_array(f,mask=dM.mask,fill_value = np.nan)
    f = f.filled()
    dM = nd.convolve1d(f,dK)

    Pcad   = round(P/keptoy.lc)
    res    = tfind.ep(tLC.t[0],dM,Pcad)
    winG   = res['win']

    # Find the first epoch after time = 0 .
    epochG = np.remainder(res['epoch'],P)
    idnn   = np.argmin( abs(epochG - epoch) )
    bwin   = winG[idnn].astype(bool)
    return bwin
