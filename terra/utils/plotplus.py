"""
Functions that supplement matplotlib.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


def one2one(*args,**kwargs):
    """
    Plot the one to one line
    """
    xl = plt.xlim()
    yl = plt.ylim()
    x = np.linspace(xl[0],xl[1],10)
    y = np.linspace(xl[0],xl[1],10)
    plt.plot(x,y,*args,**kwargs)
    

def adjust_spines(ax,spines,pad=10,smart_bounds=False):
    for loc, spine in ax.spines.items():
        if loc in spines:            
            if type(pad)==dict:
                spine.set_position(('outward',pad[loc]))
            else:
                spine.set_position(('outward',pad)) # outward by 10 points

            spine.set_smart_bounds(smart_bounds)
        else:
            spine.set_color('none') # don't draw spine
            
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def appendAxes(axlist,nplots,plotidx):
    """
    Append axes to a list
    axlist  - list of axis objects
    nplots  - total number of plots
    plotidx - which plot are we on?
    """
    axlist.append( plt.subplot(nplots,1,plotidx+1) )
    return axlist

def mergeAxes(figure):
    """
    A simple function which merges the x-axes of a figure.  Similar to
    the ``MULTIPLOT`` function in IDL.
    """
    figure.subplots_adjust(hspace=0.0001)
    axlist = figure.get_axes()
    nax = len(axlist)

    lim = (0,0)
    for i in np.arange(nax):
        curlim = axlist[i].get_xlim()
        lim = min(lim[0],curlim[0]),max(lim[1],curlim[1])

        ylim = axlist[i].get_ylim()
        yticks = axlist[i].get_yticks() 
        dtick = yticks[1] - yticks[0]
        newyticks = np.linspace(ylim[0],ylim[1],(ylim[1]-ylim[0])/dtick+1)

        # Special treatment for last axis
        if i != nax-1 :
            axlist[i].set_xticklabels('',visible=False)
            axlist[i].set_yticks(newyticks[1:])
            axlist[i].set_xlabel('')

            # Don't plot duplicate y axes
            if axlist[i].get_ylabel() == axlist[nax-1].get_ylabel():
                axlist[i].set_ylabel('')

    for ax in axlist:
        ax.set_xlim(lim)

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
    return f

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



def recMask():
    
    ax = plt.gca()
    lines = ax.get_lines()
    assert len(lines) == 1
    lines = lines[0]

    x,y = lines.get_data()

    def on_rectangle_select(event_press, event_release):
        'args the press and release events'
        x1, y1 = event_press.xdata, event_press.ydata
        x2, y2 = event_release.xdata, event_release.ydata
        print "RECT: (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2)

        if x1>x2: 
            x1, x2 = x2, x1

        if y1>y2:
            y1, y2 = y2, y1

        mask = (x>=x1) & (x<=x2) & (y>=y1) & (y<=y2) 
        return mask

    rect_select = widgets.RectangleSelector(
        ax, on_rectangle_select, drawtype='box', useblit=True,
        button=[1,], # use left button
        minspanx=5, minspany=5, spancoords='pixels', 
        # ignore rects that are too small
        );

def AddAnchored(*args,**kwargs):
    """
    Init definition: AnchoredText(self, s, loc, pad=0.4, borderpad=0.5, prop=None, **kwargs)
    Docstring:       AnchoredOffsetbox with Text
    Init docstring:
    *s* : string
    *loc* : location code
    *prop* : font property
    *pad* : pad between the text and the frame as fraction of the font
            size.
    *borderpad* : pad between the frame and the axes (or bbox_to_anchor).

    other keyword parameters of AnchoredOffsetbox are also allowed.
    """

    at = AnchoredText(*args,**kwargs)
    plt.gca().add_artist(at)

def flip(axis):
    if axis=='x':
        plt.xlim(plt.xlim()[::-1])
    if axis=='y':
        plt.ylim(plt.ylim()[::-1])
    if axis=='both':
        plt.xlim(plt.xlim()[::-1])        
        plt.ylim(plt.ylim()[::-1])


