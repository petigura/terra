"""
Python implemenation of BB.
"""

from numpy import *

def evt(tt):
    """
    Python implementation of Scargle's Bayesian blocks algorithm for
    event data.
    """

    n = len(tt)
    tt = sort(tt)
    
    # Constants
    fp_rate = .05; # Default false postive rate
    ncp_prior = 4 - log( fp_rate / ( 0.0136*n**0.478 ) );
    
    tt_top = tt[-1] + 0.5*( tt[-1] - tt[-2] ) # last tick
    tt_bot = tt[0]  - 0.5*( tt[1]  - tt[0]  ) # first tick

    mdpt = 0.5*(tt[:-1] + tt[1:]) # pts btw ticks

    # Time from tick i to the last tick
    t2end = tt_top - append(tt_bot,mdpt)    
    best,last = array([]),array([]).astype(int)

    for r in range(0,n):
        
        if r+1 < n:
            delta_offset = t2end[r+1]
        else:
            delta_offset = 0
            
        M = t2end[:r+1] - delta_offset
        M[where(M <= 0)] = inf

        # N is the number of points in B(1,i) , B(2,i) ... B(i-1,i)
        N = arange(r+1,0,-1)

        # maximum likelihood of points in B(1,i) , B(2,i) ... B(i-1,i)
        lmax = N*( log(N) - log(M) ) - ncp_prior

        # likelihood of partition
        lpart = append(0,best) + lmax

        # r* the change point that maximizes fitness of partition
        rstar = argmax(lpart)
        best,last = append( best,lpart[rstar] ),append( last,rstar )

    return last


def pt(t,x,sig):
    """
    Bayesian blocks algorithm for point measurements.  

    Use the maximum likelihold as a measure of block fitness.
    """

    # Ensure data are sequential in time.
    sidx = argsort(t)
    t   = t[sidx]
    x   = x[sidx]
    sig = sig[sidx]

    n = len(t)

    best,last,val = array([]),array([]).astype(int),array([])

    for r in range(n):

        cells = arange(r+1)

        o = [ pt_bfit(x[c:r+1],sig[c:r+1]) for c in cells]
        o = array(o) 
        Lend,valend = o[::,0],o[::,1]

        Ltot = append(0,best) + Lend - 10

        # r* the change point that maximizes fitness of partition
        rstar = argmax( Ltot)
        best  = append( best,Ltot[rstar] )
        last  = append( last,rstar )
        val   = append( val,valend[rstar] )

        if mod(r,100) ==0:
            print r

    return last,val

def pt_bfit(x,sig):
    """
    Given a data block return it's maximum likelihood.
    """

    sig2 = sig**2

    a = 0.5*sum(1. / sig2)
    b = -1.0*sum(x / sig2)
    c = 0.5*sum(x**2/sig2)

    maxl = b**2 / (4*a) - c
    maxval = -b / (2*a)

    return maxl,maxval
    
