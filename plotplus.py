import numpy as np
import matplotlib.pyplot as plt

def appendAxes(axlist,nplots,plotidx):
    """
    Append axes to a list.  Share with the first axes.
    axlist  - list of axis objects
    nplots  - total number of plots
    plotidx - which plot are we on?
    """
    
    if plotidx is 0:
        axlist.append( plt.subplot(nplots,1,plotidx+1) )
    else:
        axlist.append( plt.subplot(nplots,1,plotidx+1,sharex=axlist[0]) )

    return axlist

def mergeAxes(figure):
    """
    A simple function which merges the x-axes of a figure.
    """
    figure.subplots_adjust(hspace=0.0001)
    axlist = figure.get_axes()
    nax = len(axlist)

    for i in np.arange(nax):
        # Special treatment for last axis

        if i != nax-1 :
            axlist[i].set_xticklabels('',visible=False)
            yticks = axlist[i].get_yticks()[1:]
            axlist[i].set_yticks(yticks)
            axlist[i].set_xlabel('')

            # Don't plot duplicate y axes
            if axlist[i].get_ylabel() == axlist[nax-1].get_ylabel():
                axlist[i].set_ylabel('')

    return figure

def mergeAxesTest():
    """
    A test to see if mergeAxes is working.
    """

    x = np.linspace(0,10,100)
    y = np.sin(x)

    f = plt.figure()
    ax = []
    nplots = 3
    for i in np.arange(nplots):
        ax.append( plt.subplot(nplots,1,i+1) )
        ax[i].scatter(x,y)
        ax[i].set_xlabel('test')
        ax[i].set_ylabel('test y')

    f = mergeAxes(f)
    plt.show()


def errpt(ax,coord,xerr=None,yerr=None,**kwargs):
    """
    Overplot representitive error bar on an axis.

    ax    - axis object to manipulate and return
    coord - the coordinates of error point (in device coordinates)
            [0.1,0.1] is lower left
    xerr/yerr  : [ scalar | 2x1 array-like ] 
    """
    inv = ax.transData.inverted()
    pt = inv.transform( ax.transAxes.transform( coord ) )
    ax.errorbar(pt[0],pt[1],xerr=xerr,yerr=yerr,elinewidth=2,capsize=0,**kwargs)
    return ax

def errptTest(**kwargs):
    """
    Quick test to see if errptTest is working
    """

    ax = plt.subplot(111)
    xerr = yerr = np.array([[.1],[.2]])
    ax = errpt(ax,(.2,.2),xerr=xerr,yerr=yerr,**kwargs)
