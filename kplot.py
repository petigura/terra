"""
Plotting methods for the following classes

- Lightcurve
- Grid 
- Peak

I define these functions in a seperate module so the classes
themselves can be instanated with out import matplotlib which is not
possible on some non-interactive platforms
"""

from matplotlib.pylab import plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

import tval
import sketch
import config
import numpy as np
from numpy import ma

tprop = dict(size=10,name='monospace')


def plot_diag(pk):
    """
    Print a 1-page diagnostic plot of a given pk.
    """

    fig = plt.figure(figsize=(20,12))
    gs = GridSpec(8,10)
    axGrid     = fig.add_subplot(gs[0,0:8])
    axStack    = fig.add_subplot(gs[3: ,0:8])
    axPFAll    = fig.add_subplot(gs[1,0:8])
    axPF       = fig.add_subplot(gs[2,0:4])
    axPF180    = fig.add_subplot(gs[2,4:8],sharex=axPF,sharey=axPF)
    axScar     = fig.add_subplot(gs[0,-1])
    axSES      = fig.add_subplot(gs[1,-1])
    axSeason   = fig.add_subplot(gs[2,-1])
    axAutoCorr = fig.add_subplot(gs[3,-1])

    print pk.items()

    plt.sca(axGrid)
    plotGrid(pk)
    at = AnchoredText('Periodogram',prop=tprop, frameon=True,loc=2)
    axGrid.add_artist(at)
    axGrid.xaxis.set_ticks_position('top')
    plt.title('Period (days)')
    plt.ylabel('MES')

    plt.sca(axPFAll)
    plt.plot(pk['tPF'],pk['fmed'],',',alpha=.5)
    plt.plot(pk['bx1'],pk['by1'],'o',mew=0)
    plt.plot(pk['bx5'],pk['by5'],'.',mew=0)
    y = pk['fmed']
    axPFAll.set_ylim( (np.percentile(y,5),np.percentile(y,95) ) )

    plt.sca(axPF180)
    plotPF(pk,180)
    cax = plt.gca()
    cax.xaxis.set_visible(False)
    cax.yaxis.set_visible(False)
    at = AnchoredText('Phase Folded LC + 180',prop=tprop,frameon=True,loc=2)
    cax.add_artist(at)

    plt.sca(axPF)
    plotPF(pk,0)
    cax = plt.gca()
    cax.xaxis.set_visible(False)
    cax.yaxis.set_visible(False)
    at = AnchoredText('Phase Folded LC',prop=tprop,frameon=True,loc=2)
    cax.add_artist(at)
    df = pk.attrs['pL0'][0]**2
    plt.ylim(-5*df,3*df)

    plt.sca(axStack)
    plotSES(pk)
    plt.xlabel('Phase')
    plt.ylabel('SES (ppm)')

    plt.sca(axScar)
    sketch.scar(pk['RES'])
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)

    axSeason.set_xlim(-1,4)
    rses = pk['SES']
    axSES.plot(rses['tnum'],rses['ses']*1e6,'.')
    axSES.xaxis.set_visible(False)
    at = AnchoredText('Transit SES',prop=tprop, frameon=True,loc=2)
    axSES.add_artist(at)

    axSeason.plot(rses['season'],rses['ses']*1e6,'.')
    axSeason.xaxis.set_visible(False)
    at = AnchoredText('Season SES',prop=tprop, frameon=True,loc=2)
    axSeason.add_artist(at)

    plt.sca(axAutoCorr)
    #!pk.plotAutoCorr()
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)

    plt.gcf().text( 0.75, 0.05, pk.diag_leg() , size=10, name='monospace',
                    bbox=dict(visible=True,fc='white'))
    #plt.tight_layout()
    plt.gcf().subplots_adjust(hspace=0.01,wspace=0.01)

###########################
# Helper fuctions to plot #
###########################
def plotAutoCorr(pk):
    plt.xlabel('Displacement')
    plt.plot(pk['lag'],pk['corr'])

def plotPF(pk,ph):
    PF      = pk['lcPF%i' % ph]
    x = PF['tPF']

    try:
        plt.plot(x,PF['fPF'],',',color='k')
    except:
        pass

    try:
        plt.plot(x,PF['fit'],lw=3,color='c')
    except:
        pass

    try:
        x,y = pk['bx%i_%i'% (ph,5)],pk['by%i_%i'% (ph,5)]
        plt.plot(x,y,'o',mew=0,color='red')
    except:
        pass

def plotSES(pk):
    df = pk.attrs['pL0'][0]**2
    sketch.stack(pk.t,pk.dM*1e6,pk.attrs['P'],pk.attrs['t0'],step=3*df*1e6)
    plt.autoscale(tight=True)

def plotGrid(pk):
    x = pk['RES']['Pcad']*config.lc
    plt.plot(x,pk['RES']['s2n'])
    id = np.argsort( np.abs(x - pk.attrs['P']) )[0]
    plt.plot(x[id],pk['RES']['s2n'][id],'ro')
    plt.autoscale(tight=True)

def plotMed(pk):
    lc = pk['mqcal'][:]
    t = lc['t']
    fmed = ma.masked_array(pk['fmed'][:],lc['fmask'])
    P = pk.attrs['P']
    t0 = pk.attrs['t0']
    df = pk.attrs['df']
    sketch.stack(t,fmed*1e6,P,t0,step=5*df)
    plt.autoscale(tight=True)

def morton(pk):
    """
    Print a 1-page diagnostic plot of a given pk.
    """

    fig = plt.figure(figsize=(20,12))
    gs = GridSpec(4,4)

    axPF       = fig.add_subplot(gs[0,0:2])
    axPF180    = fig.add_subplot(gs[0,2:],sharex=axPF,sharey=axPF)
    axStack    = fig.add_subplot(gs[1:,:])

    plt.sca(axPF180)
    plotPF(pk,180)
    cax = plt.gca()
    cax.xaxis.set_visible(False)
    cax.yaxis.set_visible(False)
    at = AnchoredText('Phase Folded LC + 180',prop=tprop,frameon=True,loc=2)
    cax.add_artist(at)

    plt.sca(axPF)
    plotPF(pk,0)
    cax = plt.gca()
    at = AnchoredText('Phase Folded LC',prop=tprop,frameon=True,loc=2)
    cax.add_artist(at)
    plt.xlabel('t - t0 (days)')
    plt.ylabel('flux')
    df = pk.attrs['df']*1e-6
    plt.ylim(-5*df,3*df)

    plt.sca(axStack)
    plotMed(pk)
    plt.xlabel('phase')
    plt.ylabel('flux (ppm)')

    #plt.gcf().text( 0.85, 0.05, pk.diag_leg() , size=10, name='monospace',
    #                bbox=dict(visible=True,fc='white'))
    #plt.tight_layout()
    plt.gcf().subplots_adjust(hspace=0.21,wspace=0.05,left=0.05,right=0.99,bottom=0.05,top=0.99)
