"""
An interface to the matlab Bayesian Blocks code that handles event data.
"""


from mlabwrap import mlab
from numpy import *
import matplotlib.pylab as plt
from numpy.random import rand,randn

def stest():
    """
    Jeff Scargle's Test case.
    """

    nsig = 512
    nbkg = 1024

    bkg = rand(1,nbkg)
    sig = 0.4 + 0.05*randn(1,nsig)
    
    x = append(bkg,sig)
    x = sort(x)
    bb = mlab.find_blocks(x)
    
    plotbb(x,bb)


def etest():
    """
    My Test case.
    """

    x = append(linspace(0,1,100), linspace(1,2,10 )  )
    bb = mlab.find_blocks(x)
    
    plotbb(x,bb)

def exp():
    
    
    nsig = 512
    nbkg = 1024

    bkg = rand(1,nbkg)
    sig = 0.4 + 0.05*randn(1,nsig)
    
    x = append(bkg,sig)
    x = sort(x)
    bb = mlab.find_blocks(x)
    
    plotbb(x,bb)


def plotbb(x,bb,**kwargs):
    """
    Plot a histogram of the data and show where Bayesian Blocks wants
    to put the change points.
    """

    f = plt.gcf() 
    f.clf()
    ax = f.add_subplot(111)
    ax.hist(x,**kwargs)
    
    # Matlab returns matricies, vlines expects arrays
    bb = (bb.flatten()).astype(int)
    chpts = x[bb]
    
    ylim = ax.get_ylim()
    ax.vlines(chpts,0,ylim[1])
    plt.show()
    
def val2tt(x,y):
    """
    convert an x y series into an tt series
    """

    usamp = 1000

    tt = array([])
    ymax  = max(y)
    n = len(x)
    
    for i in range(n-1):
        nhits = float(y[i])/ymax*usamp
        t = linspace(x[i],x[i+1],nhits)
        tt = append(tt,t)


    return tt

    


