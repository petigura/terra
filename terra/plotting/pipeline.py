"""Plotting of pipeline objects"""

import pandas as pd
from matplotlib import pylab as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib import ticker

from kplot import wrapHelp,yticksppm,tprop,bbox,annkw
from terra import tfind
from .. import tval

from matplotlib.ticker import MaxNLocator
import numpy as np
pd.set_eng_float_format(accuracy=3,use_eng_prefix=True)
plt.rc('axes',color_cycle=['RoyalBlue','Tomato'])
plt.rc('font',size=8)

def diagnostic(pipe):
    """
    Print a 1-page diagnostic plot of a given pipeline object

    Args:
        pipe (terra.pipeline.Pipeline) : pipeline object
    """
    fig = plt.figure(figsize=(20,12))
    
    gs_kw = dict(left=0.04,right=0.99)

    # Provision the upper plots
    gs1 = GridSpec(3,10)
    gs1.update(hspace=0.05, wspace=0.5,bottom=0.65,top=0.95, **gs_kw)
    axPeriodogram  = fig.add_subplot(gs1[0,0:8])
    axAutoCorr = fig.add_subplot(gs1[0,8])
    
    # These 4 plots all share an xaxis
    ax_transit = fig.add_subplot(gs1[1,0:2])
    ax_transit_zoom = fig.add_subplot(gs1[2,0:2],sharex=ax_transit)
    ax_phasefold_180 = fig.add_subplot(gs1[1,2:4],sharex=ax_transit)
    ax_phasefold_secondary = fig.add_subplot(gs1[2,2:4],sharex=ax_transit)

    ax_phasefold = fig.add_subplot(gs1[1:,4:8])
    axTransitTimes = fig.add_subplot(gs1[1,8])
    axTransitRp = fig.add_subplot(gs1[2,8])

    # Provision the lower plots
    gs2 = GridSpec(1,5)
    gs2.update(hspace=0.05,wspace=0.2,bottom=0.05,top=0.6,**gs_kw)
    axSES = fig.add_subplot(gs2[0 ,0:4])
    axTransitStack = fig.add_subplot(gs2[0 ,4])

    # Periodogram
    plt.sca(axPeriodogram)
    periodogram(pipe)
    ax = plt.gca()
    at = AnchoredText('Periodogram',prop=tprop, frameon=True,loc=2)
    ax.add_artist(at)

    # Auto correlation plot, and header text, defined to the right
    plt.sca(axAutoCorr)
    ax = plt.gca()
    autocorr(pipe)
    AddAnchored("ACF",prop=tprop,frameon=True,loc=2)    
    text = header_text(pipe)
    plt.text(
        1.1, 1.0, text, family='monospace', size='large', 
        transform=ax.transAxes, va='top',
    )

    # Transit times, holding depth constant
    plt.sca(axTransitTimes)
    transits(pipe,mode='transit-times')

    # Transit radius ratio
    plt.sca(axTransitRp)
    transits(pipe,mode='transit-rp')

    # Phasefolded photometry 
    plt.sca(ax_transit)
    phasefold_transit(pipe,mode='transit')
    plt.ylabel('Flux')

    # Zoomed phase folded photometry
    plt.sca(ax_transit_zoom)
    phasefold_transit(pipe,mode='transit-zoom')

    # Light curve 180 degrees out of phase
    plt.sca(ax_phasefold_180)
    phasefold_shifted(pipe, 0.5)

    # Phase-folded light curve at the most significant secondary eclipse
    plt.sca(ax_phasefold_secondary)
    phasefold_shifted(pipe, pipe.se_phase)

    # Phase-folded all phases
    plt.sca(ax_phasefold)
    lc, lcbin = phasefold_shifted(pipe, 0, xdata='phase')
    yl = [lcbin.fmed.min(),lcbin.fmed.max()]
    span = lcbin.fmed.ptp()
    yl[0] -= 0.25 * span
    yl[1] += 0.25 * span
    plt.ylim(*yl)
    plt.autoscale(axis='x',tight=True)
    plt.xlabel('Phase')
    plt.ylabel('Flux')

    # Sinngle Event Statistic Stack
    plt.sca(axSES)
    mad = pipe.lc.f.abs().median()
    ystep = 3 * 1.6 * mad
    ystep = max(pipe.fit_rp**2 * 1.0, ystep)

    single_event_statistic_stack(pipe, ystep=ystep)
    pipe.se_phase 
    plt.axvline(0, alpha=.1,lw=10,color='m',zorder=1)

    # Sinngle Event Statistic Stack
    plt.sca(axTransitStack)
    transit_stack(pipe,ystep=ystep)


    axL = [
        ax_transit, ax_transit_zoom, ax_phasefold_180, ax_phasefold, 
        axTransitTimes, axTransitRp
    ]
    for ax in axL:
        ax.yaxis.set_major_locator(MaxNLocator(4))

    axL = [
        ax_transit, ax_transit_zoom, ax_phasefold_180, ax_phasefold, 
        axTransitTimes, axTransitRp, axSES, axTransitStack,
    ]
    for ax in fig.get_axes():
        ax.grid()

def header_text(pipe):
    
    text = """\
name  {starname}
P     {fit_P:.3f}
t0    {fit_t0:.2f}
tdur  {fit_tdur:.2f} 
SNR   {grid_s2n:.2f}
""".format(**pipe.header.value)
    return text

def periodogram(pipe):
    """Create a plot of the periodogram with nice labels.

    Highlights highest SNR point and also overplots harmonics
    """
    pgram = pipe.pgram


    # Nicely formatted periodogram ticks
    xt = [ 
        0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 20, 30, 40, 50, 60, 70, 80, 90,
        100, 200, 300, 400, 500, 600, 700, 800, 900,
    ]
    
    grid_P = pipe.header.ix['grid_P'].value
    grid_s2n = pipe.header.ix['grid_s2n'].value

    plt.semilogx()
    peak = pgram.sort('s2n').iloc[-1]

    # Plot periogram, peak, and harmonics
    plt.plot(pgram.P, pgram.s2n)
    plt.plot(grid_P, grid_s2n, 'ro')
    harm = np.array([2,4]).astype(float)
    harm = np.hstack([1,1/harm,harm]) 
    harm *= grid_P
    for h in harm:
        plt.axvline(h,lw=6,alpha=0.2,color='r',zorder=0)

    # Label plot
    ax = plt.gca()
    plt.autoscale(axis='x',tight=True)
    xl = plt.xlim() # Save away the tight-fitting limits
    plt.xticks(xt, xt) # Add the nicely formatted ticks
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top') 
    plt.xlim(*xl) # Reset the plot limits.
    plt.xlabel('Period [days]')
    plt.ylabel('SNR')
    plt.draw()

def autocorr(pipe):
    """Simple plot of auto correlation"""
    auto = pipe.auto
    plt.plot(auto.t_shift,auto.autocorr)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)



#def phaseFold(dv,ph,diag=False,zoom=False):
#    PF = getattr(dv,'lcPF%i' % ph )
#    x  = PF['tPF']
#
#    @handelKeyError
#    def plot_phase_folded():
#        if zoom==False:
#            plot(x,PF['f'],'.',ms=2,color='k')
#        bPF = getattr(dv,'blc30PF%i' % ph)
#        t   = bPF['tb']
#        f   = bPF['med']
#        plot(t,f,'+',ms=5,mew=1.5,color='Chartreuse')
#
#    @handelKeyError
#    def plot_fit():
#        trans  = dv.trans
#        plot(trans.t,trans.fit,'--',color='Tomato',lw=3,alpha=1)
#
#    plot_phase_folded()
#    if ph==0:
#        plot_fit()
#        title='ph = 0'
#        xl   ='t - t0 [days]'
#    else:
#        title='ph = 180'
#        xl   ='t - (t0 + P/2) [days]'
#    
#    if dv.mean < 1e-4:
#        ylim(-5*dv.mean,5*dv.mean)
#
#    if zoom==True:
#        autoscale(axis='y')
#    if diag:
#        AddAnchored(title,prop=tprop,frameon=False,loc=4)
#    
#    autoscale('x')    
#    xlabel(xl)

def roblims(x,p,fac):
    """
    Robust Limits

    Return the robust limits for a plot

    Parameters
    ----------
    x  : input array
    p : percentile (compared to 50) to use as lower limit
    fac : add some space
    """
    plo, pmed, phi = np.percentile(x,[p,50,100-p])
    span = phi - plo
    lim = plo - span*fac, phi + span*fac
    return lim

def phasefold_transit(pipe, mode='transit'):
    """
    Plot a phasefolded plot of the photometry

    Args: 
        pipe (terra.pipeline.Pipeline) : pipeline object
        mode (Optional[str]) : {'transit','transit-zoom',transit-180',}
    """

    lc = pipe.lcdt
    lcfit = pipe.lcfit
    lcbin = pipe.lcdtpfbin
    if mode=='transit':
        yl = roblims(lc.f, 5 , 0.5)
    if mode=='transit-zoom':        
        depth = pipe.fit_rp**2
        yl = (-2.0 * depth, depth)

    xl = lc.t_phasefold.min(), lc.t_phasefold.max()
    label=mode

    # plot the unbinned data points
    plt.plot(lc.t_phasefold,lc.f,'o',color='k',alpha=0.8,ms=2,mew=0)

    # plot the binned data points
    if np.mean(lcbin.fcount) > 3:
        lcbin = lcbin.query('fcount > 0')
        plt.errorbar(
            lcbin.t_phasefold, lcbin.fmed, yerr=lcbin. ferr, fmt='o', ms=5, 
            mew=0, capsize=0)

    if type(lcfit) is not type(None):
        lcfit = lcfit.sort('t_phasefold')
        plt.plot(lcfit.t_phasefold,lcfit.f,lw=2,color='Tomato',alpha=0.9)

    plt.xlim(*xl) # reset the x-limits 
    plt.ylim(*yl) # reset the x-limits 
    AddAnchored(label,prop=tprop,frameon=True,loc=3)
    plt.xlabel('Time Since Transit (days)')

def phasefold_shifted(pipe, phase_shift, xdata='t_phasefold'):
    """
    Plot a phasefolded plot of the photometry

    Args: 
        pipe (terra.pipeline.Pipeline) : pipeline object
        phase_shift (float) : phase to shift points by
    """
    starting_phase = -0.25
    phase_shift = float(phase_shift)
    lc = pipe.lc.copy()
    lc = lc[~lc.fmask]

    t0 = pipe.fit_t0 + phase_shift * pipe.fit_P
    lc = tval.add_phasefold(
        lc, lc.t, pipe.fit_P, t0, starting_phase=starting_phase
    )
    P = pipe.fit_P
    dt = pipe.dt
    
    n_phase_bins = np.round(P / dt)
    t_start = starting_phase * P
    t_stop = (starting_phase + 1) * P
    t_phasefold_bins = np.linspace(t_start, t_stop, n_phase_bins)
    lcbin = tval.bin_phasefold(lc, t_phasefold_bins)
    lcbin = tval.add_phasefold(
        lcbin, lcbin.t_phasefold, pipe.fit_P, 0, starting_phase=starting_phase
    )    

    # plot the unbinned data points
    plt.plot(lc[xdata],lc.f,'o',color='k',alpha=0.8,ms=2,mew=0)
    lcbin = lcbin.query('fcount > 0')
    plt.errorbar(
        lcbin[xdata], lcbin.fmed, yerr=lcbin. ferr, fmt='o', ms=5, 
        mew=0, capsize=0, lw=0)

    xl = (-3.0 * pipe.fit_tdur, 3.0 * pipe.fit_tdur )
    fmed = lcbin[lcbin[xdata].between(*xl)].fmed
    yl = roblims(fmed, 5 , 0.5)
    plt.xlim(*xl)
    plt.ylim(*yl)

    label = "ph = {:.2f}".format(phase_shift)
    AddAnchored(label,prop=tprop,frameon=True,loc=3)
    plt.xlabel('Time Since Reference Phase (days)')
    return lc, lcbin

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

def single_event_statistic_stack(pipe, ystep=None):
    if ystep==None:
        depth = pipe.fit_rp**2
        ystep = depth*2

    twd = int(pipe.grid_tdur / pipe.dt)
    lc = pipe.lc[['t']].copy() # create a temp lightcurve
    t_phasefold, phase, cycle = tval.phasefold(
        lc.t, pipe.fit_P, pipe.fit_t0, starting_phase=-0.25
    )
    lc['t_phasefold'] = t_phasefold
    lc['phase'] = phase
    lc['cycle'] = cycle
    fm = pipe._get_fm()
    lc['dM'] = tfind.mtd(fm,twd)

    yshift = 0
    g = lc.groupby('cycle')
    for cycle, idx in g.groups.iteritems():
        lc_single = lc.ix[idx]
        lc_single = lc_single.sort('phase')
        plt.plot(lc_single.phase,lc_single.dM + yshift,ms=4)
        yshift -= ystep

    plt.xlim(-0.25,0.75)
    plt.xlabel('Phase')
    plt.ylabel('Transit Depth (Box Width = {} cadences)'.format(twd) )
    plt.axvline(0, alpha=.1,lw=10,color='m',zorder=1)
    plt.axvline(0.5, alpha=.1,lw=10,color='m',zorder=1)
    AddAnchored("SES Stack",prop=tprop,frameon=True,loc=2)

def transit_stack(pipe, ystep=None):
    if ystep==None:
        depth = pipe.fit_rp**2
        ystep = depth*2

    lcdt = pipe.lcdt
    lcfit = pipe.lcfit.sort('t_phasefold')
    g = pipe.lcdt.groupby('transit_id')

    yshift = 0
    for transit_id, idx in g.groups.iteritems():
        lc = pipe.lcdt.ix[idx]
        plt.plot(lc.t_phasefold,lc.f + yshift,'o-k',ms=5)
        plt.plot(lcfit.t_phasefold,lcfit.f + yshift,'-',lw=2)
        yshift -= ystep

    xl = lcdt.t_phasefold.min(), lcdt.t_phasefold.max()
    plt.xlim(*xl)
    plt.xlabel('Time Since Mid Transit (days)')
    AddAnchored("Transit Stack",prop=tprop,frameon=True,loc=2)

def transits(pipe,mode='transit-times'):
    _transits = pipe.transits
    if mode=='transit-times':
        y = _transits.omc
        yerr = _transits.ut0
        ylabel = 'O - C'
    if mode=='transit-rp':
        y = _transits.rp
        yerr = _transits.urp
        ylabel = 'Rp/Rstar'

    x = _transits.transit_id
    xlabel = 'transit-id'
    xl = x.min() -1, x.max()+1 

    plt.errorbar(x, y, yerr=yerr, fmt='o', ms=5, mew=0, capsize=0)
    plt.xlim(*xl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    AddAnchored(mode,prop=tprop,frameon=True,loc=2)


def AddAnchored(*args,**kwargs):
    # Hack to get rid of warnings
    for k in 'ha va'.split():
        if kwargs['prop'].has_key(k):
            kwargs['prop'].pop(k)

    at = AnchoredText(*args,**kwargs)
    plt.gca().add_artist(at)
