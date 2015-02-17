"""
Functions to supplement numpy.
"""

import numpy as np

def binavg(x,y,bins):
    """
    Computes the average value of y on a bin by bin basis.

    x - array of x values
    y - array 
    bins - array of bins in x

    x = array([1,2,3,4])
    y = array([1,3,2,4])
    bins = array([0,2.5,5])
    
    returns
    
    array([1.25,3.75]),array([2.,3.])

    Notes:

    Benchmarked this over a similar fucntion that loops over where statements
    While the performance is comparable for nbins ~ 10, as one goes to higher
    number of bins, doing the sorting outside the loop makes a *huge* difference
    """

    #make the arrays monotonically increasing in x
    sarg = np.argsort(x)
    x = x[sarg]
    y = y[sarg]

    # for each point compute the index of the bin that it falls into.
    idx = np.digitize(x,bins) 

    # compute the midpoint of each bin.
    binx = bins[:-1] + (bins[1:]-bins[:-1])/2
    nbin = len(binx)
    biny = np.empty(nbin,dtype=y.dtype)

    #number of points in each bin
    nidx = np.bincount(idx)[1:]

    lslice = 0 # the left most slice
    for i in range(nbin):
        # Compute the average value in each bin
        rslice = lslice + nidx[i]
        biny[i] = np.mean(y[lslice:rslice]) 
        lslice += nidx[i] # move left slice up

    return binx,biny

def hbinavg(x,y,bins):
    """
    Computes the average value of y on a bin by bin basis.

    x - array of x values
    y - array 
    bins - array of bins in x

    x = array([1,2,3,4])
    y = array([1,3,2,4])
    bins = array([0,2.5,5])
    
    returns
    
    array([1.25,3.75]),array([2.,3.])

    Notes:
    This function is simplier than previous binavg and uses built in
    numpy functionality.  For situtations with few (<10) bins, binavg
    performs better.  Above that, hbinavg blows it away.

    ==== ======= ======= ======= ========
    n    nbins   binavg  hbinavg speed up 
    ==== ======= ======= ======= ========
    4    2       48.8 us 236 us  0.2
    10   10      111 us  236 us  0.5
    1e3  1e3     8.16 ms 441 us  20
    1e4  1e4     108 ms  2.55 ms 40
    1e5  1e5     3.92 s  42.5 ms 100
    1e3  10      147 us  280 us  0.5
    1e3  1e2     8.04 ms 5.19 ms 1.5
    1e6  1e5     32.3 s  304 ms  100
    =====================================
    """

    binx = bins[:-1] + (bins[1:] - bins[:-1])/2.
    bsum = ( np.histogram(x,bins=bins,weights=y) )[0]
    bn   = ( np.histogram(x,bins=bins) )[0]
    biny = bsum/bn

    return binx,biny
    
