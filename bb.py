"""
Bayesian Blocks Transit Finder
"""

def get_blocks(x,last,val):
    """
    Just return the blocks.

    If a timeseries only has one block, all elements will be equal.
    """

    cp = unique(last)
    ncp  = len(cp)

    n = len(x)
    idxlo = cp                    # index of left side of region
    idxhi = append(cp[1:],n)-1 # index of right side of region 

    for i in range(ncp):
        x[ idxlo[i]:idxhi[i]+1 ] = val[ idxhi[i] ]

    return x
