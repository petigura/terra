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

tprop = dict(size=10,name='monospace')

class Peak(tval.Peak):
    def plot_diag(self):
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

        plt.sca(axGrid)
        self.plotGrid()
        at = AnchoredText('Periodogram',prop=tprop, frameon=True,loc=2)
        axGrid.add_artist(at)
        axGrid.xaxis.set_ticks_position('top')
        plt.title('Period (days)')
        plt.ylabel('MES')

        plt.sca(axPFAll)
        plt.plot(self['tPF'],self['fmed'],',',alpha=.5)
        plt.plot(self['bx1'],self['by1'],'o',mew=0)
        plt.plot(self['bx5'],self['by5'],'.',mew=0)
        y = self['fmed']
        axPFAll.set_ylim( (np.percentile(y,5),np.percentile(y,95) ) )

        plt.sca(axPF180)
        self.plotPF(180)
        cax = plt.gca()
        cax.xaxis.set_visible(False)
        cax.yaxis.set_visible(False)
        at = AnchoredText('Phase Folded LC + 180',prop=tprop,frameon=True,loc=2)
        cax.add_artist(at)

        plt.sca(axPF)
        self.plotPF(0)
        cax = plt.gca()
        cax.xaxis.set_visible(False)
        cax.yaxis.set_visible(False)
        at = AnchoredText('Phase Folded LC',prop=tprop,frameon=True,loc=2)
        cax.add_artist(at)
        df = self.attrs['pL0'][0]**2
        plt.ylim(-5*df,3*df)

        plt.sca(axStack)
        self.plotSES()
        plt.xlabel('Phase')
        plt.ylabel('SES (ppm)')

        plt.sca(axScar)
        sketch.scar(self.res)
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)

        axSeason.set_xlim(-1,4)
        rses = self['SES']
        axSES.plot(rses['tnum'],rses['ses']*1e6,'.')
        axSES.xaxis.set_visible(False)
        at = AnchoredText('Transit SES',prop=tprop, frameon=True,loc=2)
        axSES.add_artist(at)

        axSeason.plot(rses['season'],rses['ses']*1e6,'.')
        axSeason.xaxis.set_visible(False)
        at = AnchoredText('Season SES',prop=tprop, frameon=True,loc=2)
        axSeason.add_artist(at)

        plt.sca(axAutoCorr)
        self.plotAutoCorr()
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        
        plt.gcf().text( 0.75, 0.05, self.diag_leg() , size=10, name='monospace',
                        bbox=dict(visible=True,fc='white'))
        plt.tight_layout()
        plt.gcf().subplots_adjust(hspace=0.01,wspace=0.01)

    ###########################
    # Helper fuctions to plot #
    ###########################
    def plotAutoCorr(self):
        plt.xlabel('Displacement')
        plt.plot(self['lag'],self['corr'])

    def plotPF(self,ph):
        PF      = self['lcPF%i' % ph]
        x,y,yfit = PF['tPF'],PF['fPF'],PF['fit']
        plt.plot(x,y,',',color='k')
        plt.plot(x,yfit,lw=3,color='c')        

        x,y = self['bx%i_%i'% (ph,5)],self['by%i_%i'% (ph,5)]
        plt.plot(x,y,'o',mew=0,color='red')

    def plotSES(self):
        df = self.attrs['pL0'][0]**2
        sketch.stack(self.t,self.dM*1e6,self.P,self.t0,step=3*df*1e6)
        plt.autoscale(tight=True)

    def plotGrid(self):
        x = self.res['Pcad']*config.lc
        plt.plot(x,self.res['s2n'])
        id = np.argsort( np.abs(x - self.P) )[0]
        plt.plot(x[id],self.res['s2n'][id],'ro')
        plt.autoscale(tight=True)
