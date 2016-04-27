"""
Functions to facilitate the plotting of tval objects
"""
import pandas as pd
from matplotlib.pylab import *
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from kplot import wrapHelp,yticksppm,tprop,bbox,annkw
from .. import transit_model as tm
from .. import tval
from .. import config
from matplotlib.ticker import MaxNLocator


pd.set_eng_float_format(accuracy=3,use_eng_prefix=True)
rc('axes',color_cycle=['RoyalBlue','Tomato'])
rc('font',size=8)

def AddAnchored(*args,**kwargs):
    # Hack to get rid of warnings
    for k in 'ha va'.split():
        if kwargs['prop'].has_key(k):
            kwargs['prop'].pop(k)

    at = AnchoredText(*args,**kwargs)
    gca().add_artist(at)

def diag(dv,tpar=False):
    """
    Print a 1-page diagnostic plot of a given h5.
    
    Right now, we recompute the single transit statistics on the
    fly. By default, we show the highest SNR event. We can fold on
    arbitrary ephmeris by setting the tpar keyword.

    Parameters
    ----------
    h5   : h5 file after going through terra.dv
    tpar : Dictionary with alternate ephemeris specified by:
           Pcad - Period [cadences] (float)
           t0   - transit epoch [days] (float)
           twd  - width of transit [cadences]
           mean - depth of transit [float]
           noise 
           s2n
    """
    fig = figure(figsize=(20,12))
    gs = GridSpec(8,10)

    # Top row
    axPeriodogram  = fig.add_subplot(gs[0,0:8])
    axAutoCorr = fig.add_subplot(gs[0,8])

    # Second row
    axPF       = fig.add_subplot(gs[1,0:2])
    axPFzoom   = fig.add_subplot(gs[1,2:4],sharex=axPF,)
    axPF180    = fig.add_subplot(gs[1,4:6],sharex=axPF)
    axPFSec    = fig.add_subplot(gs[1,6:8],sharex=axPF)
    axSingSES  = fig.add_subplot(gs[1,-2])

    # Last row
    axStack        = fig.add_subplot(gs[2:8 ,0:8])
    axStackZoom    = fig.add_subplot(gs[2:8 ,8:])

    # Top row
    sca(axPeriodogram)
    periodogram(dv)

    sca(axAutoCorr)
    autocorr(dv)
    AddAnchored("ACF",prop=tprop,frameon=True,loc=2)    

    d = dict([k,getattr(dv,k)] for k in dv.attrs_keys)
    s = """
P     %(P).3f 
t0    %(t0).2f 
tdur  %(tdur).2f 
SNR   %(s2n).2f
grass %(grass).2f""" % d
    text(1.2,0,s,family='monospace',size='large',transform=gca().transAxes)

    # Second row
    sca(axPF)
    phaseFold(dv,0,diag=True)
    ylabel('Flux')
    AddAnchored("Phased",prop=tprop,frameon=True,loc=3)

    sca(axPFzoom)
    phaseFold(dv,0,diag=True,zoom=True)
    AddAnchored("Phased Zoom",prop=tprop,frameon=True,loc=3)

    sca(axPF180)
    phaseFold(dv,180,diag=True)
    AddAnchored("Phased 180",prop=tprop,frameon=True,loc=3)

    sca(axPFSec)
    secondary_eclipse(dv)
    AddAnchored("Secondary\nEclipse",prop=tprop,frameon=True,loc=3)

    for ax in [axPF,axPFzoom,axPF180,axPFSec]:
        ax.grid()
        ax.yaxis.set_major_locator(MaxNLocator(4))

    sca(axSingSES)
    single_event_statistic(dv)

    # Bottom row
    sca(axStack)
    single_event_statistic_stack(dv)
    xlabel('Phase')
    AddAnchored("SES Stack",prop=tprop,frameon=True,loc=2)

    sca(axStackZoom)
    if dv.num_trans < 20: 
        transit_stack(dv)
        xlabel('t-t0')
        AddAnchored("Transit Stack",prop=tprop,frameon=True,loc=2)
    else:
        AddAnchored("Transit Stack\nToo Many Transits",
                    prop=tprop,frameon=True,loc=2)
        gca().xaxis.set_visible(False)
        gca().yaxis.set_visible(False)
        pass

    fig.set_tight_layout(True)
    fig.subplots_adjust(hspace=0.4)

def periodogram(dv):
    cax = gca()
    res,lc = dv.pgram,dv.lc
    x = res['Pcad']*config.lc
    y = res['s2n']

    semilogx()
    xt = [0.1,0.2,0.5,1,2,5,10,20,50,100,200,400]

    xt = [ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9] + \
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] +\
         [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90] +\
         [  0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    xticks(xt,xt)
    at = AnchoredText('Periodogram',prop=tprop, frameon=True,loc=2)
    cax.add_artist(at)
    ylabel('SNR')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top') 
    xlabel('Period [days]')
    
    plot(x,y)
    id = np.argsort( np.abs(x - dv.P))[0]
    plot(x[id],y[id],'ro')
    autoscale(axis='x',tight=True)
    xl = xlim()

    harm = np.array([2,4]).astype(float)
    harm = np.hstack([1,1/harm,harm]) 
    harm *= dv.P 
    for h in harm:
        axvline(h,lw=6,alpha=0.2,color='r',zorder=0)
    xlim(*xl)

def autocorr(dv):
    xlabel('Displacement')
    plot(dv.lag,dv.corr)
    gca().xaxis.set_visible(False)
    gca().yaxis.set_visible(False)

def phaseFold(dv,ph,diag=False,zoom=False):
    PF = getattr(dv,'lcPF%i' % ph )
    x  = PF['tPF']

    @handelKeyError
    def plot_phase_folded():
        if zoom==False:
            plot(x,PF['f'],'.',ms=2,color='k')
        bPF = getattr(dv,'blc30PF%i' % ph)
        t   = bPF['tb']
        f   = bPF['med']
        plot(t,f,'+',ms=5,mew=1.5,color='Chartreuse')

    @handelKeyError
    def plot_fit():
        trans  = dv.trans
        plot(trans.t,trans.fit,'--',color='Tomato',lw=3,alpha=1)

    plot_phase_folded()
    if ph==0:
        plot_fit()
        title='ph = 0'
        xl   ='t - t0 [days]'
    else:
        title='ph = 180'
        xl   ='t - (t0 + P/2) [days]'
    
    if dv.mean < 1e-4:
        ylim(-5*dv.mean,5*dv.mean)

    if zoom==True:
        autoscale(axis='y')
    if diag:
        AddAnchored(title,prop=tprop,frameon=False,loc=4)
    
    autoscale('x')    
    xlabel(xl)

def handelKeyError(func):
    """
    Cut down on the number of try except statements
    """
    def handelProblems():
        try:
            func()
        except KeyError:
            print "%s: KeyError" %  func.__name__ 
    return handelProblems


def secondary_eclipse(dv):
    """
    Plot secondary eclipse
    """
    plot(dv.lcPF_SE['tPF'],dv.lcPF_SE['f'])
    title = 'ph = %.1f' % dv.ph_SE
    AddAnchored(title,prop=tprop, frameon=False,loc=4)
    xl   ='t - (t0 + %.1f) [days]' % dv.t0shft_SE
    xlabel(xl)

def single_event_statistic(dv):    
    cax = gca()
    rses = dv.SES
    plot(rses['tnum'],rses['ses']*1e6,'_',mfc='k',mec='k',ms=4,mew=2)
    cax.xaxis.set_visible(False)
    AddAnchored('Transit SES',prop=tprop, frameon=False,loc=3)
    xl = xlim()
    xlim(xl[0]-1,xl[-1]+1)
    yl = ylim()
    ylim(-100,yl[1])


def single_event_statistic_stack(dv):
    d = dict(time=False)
    wrapHelp(dv,dv.t,dv.dM*1e6,d)
    ylabel('SES (ppm) [%.1f hour]' % (dv.twd/2.0))
    axvline(0, alpha=.1,lw=10,color='m',zorder=1)
    axvline(.5,alpha=.1,lw=10,color='m',zorder=1)

def transit_stack(dv):
    d = dict(time=True)

    res,lc = dv.pgram,dv.lc
    ym = ma.masked_array(lc['fcal']*1e6,lc['fmask'])
    wrapHelp(dv,lc['t'],ym,d,time=True)
    ylabel('SES (ppm)')

    # With of transit view units of twd
    xl = np.array([-3.0,3.0])
    xl = xl* dv.twd * config.lc
    xlim(*xl)
    axvline(0, alpha=.1,lw=10,color='m',zorder=1)

    gca().yaxis.set_visible(False)

