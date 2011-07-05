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
