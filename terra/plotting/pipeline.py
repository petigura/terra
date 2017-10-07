"""Plotting of pipeline objects"""
import collections
import traceback
import sys
import textwrap

import numpy as np
import pandas as pd

from matplotlib import pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from kplot import tprop
from .. import tfind
from .. import tval

pd.set_eng_float_format(accuracy=3,use_eng_prefix=True)
plt.rc('axes',color_cycle=['RoyalBlue','Tomato'])
plt.rc('font',size=8)
def print_traceback(f):
    """
    Decorator so that we can fail gracefully from a plotting mishap
    """
    def wrapper_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            ax = plt.gca()
            error = traceback.format_exc()
            print error
            error = textwrap.fill(error,50)
            ax.text(0, 1, error, transform=ax.transAxes, va='top')
    return wrapper_function

@print_traceback
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
    gs1.update(hspace=0.2, wspace=0.6,bottom=0.65,top=0.95, **gs_kw)
    axPeriodogram  = fig.add_subplot(gs1[0,0:8])
    axACF = fig.add_subplot(gs1[0,8])
    
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
    plt.sca(axACF)
    ax = plt.gca()
    autocorr(pipe)
    AddAnchored("ACF",prop=tprop,frameon=True,loc=2)    

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

    # Single Event Statistic Stack
    plt.sca(axSES)
    mad = pipe.lc.f.abs().median()
    ystep = 3 * 1.6 * mad
    ystep = max(pipe.fit_rp**2 * 1.0, ystep)

    single_event_statistic_stack(pipe, ystep=ystep)
    pipe.se_phase 
    plt.axvline(0, alpha=.1,lw=10,color='m',zorder=1)

    # Single Event Statistic Stack
    plt.sca(axTransitStack)
    transit_stack(pipe,ystep=ystep)

    axL = [
        ax_transit, ax_transit_zoom, ax_phasefold_180, ax_phasefold_secondary,
        ax_phasefold, axTransitTimes, axTransitRp
    ]
    for ax in axL:
        ax.yaxis.set_major_locator(MaxNLocator(4))

    axL = [
        ax_transit, ax_transit_zoom, ax_phasefold_180, ax_phasefold, 
        axTransitTimes, axTransitRp, axSES, axTransitStack,
    ]
    for ax in fig.get_axes():
        ax.grid()

    plt.sca(axACF)
    text = header_text(pipe)
    plt.text(
        1.1, 1.0, text, family='monospace', size='large', 
        transform=axACF.transAxes, va='top',
    )




def header_text(pipe):
    header = dict(**pipe.header.value)
    header['depth_ppm'] = header['fit_rp']**2 * 1e6

    fmtd = collections.OrderedDict()
    fmtd['starname'] = 's'
    fmtd['candidate'] = 'd'
    fmtd['grid_s2n'] = '.2f'
    fmtd['fit_P'] = '.3f'
    fmtd['fit_t0'] = '.2f'
    fmtd['fit_rp'] = '.2f'
    fmtd['fit_tdur'] = '.2f'
    fmtd['fit_b'] = '.2f'
    fmtd['fit_rchisq'] = '.2f'
    fmtd['depth_ppm'] = '.0f'
    fmtd['autor'] = '.2f'
    fmtd['se_s2n'] = '.1f'
    fmtd['se_t0'] = '.2f'
    fmtd['se_phase'] = '.2f'
        
    text = ""

    for key, fmt in fmtd.iteritems():
        if header.has_key(key):
            value = header[key]
            text += "{:13s}{:{}}\n".format(key,value,fmt)
        else:
            text += "{:13s}{:{}}\n".format(key,'none','s')

    return text

@print_traceback
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
    peak = pgram.sort_values('s2n').iloc[-1]

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

@print_traceback
def autocorr(pipe):
    """Simple plot of auto correlation"""
    auto = pipe.auto
    plt.plot(auto.t_shift,auto.autocorr)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)

@print_traceback
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
    plt.plot(lc.t_phasefold,lc.f,'o',color='k',markersize=3,alpha=0.8,mew=0)

    # plot the binned data points
    if lcbin.query('fcount > 0').fcount.mean() > 4:
        lcbin = lcbin.query('fcount > 0')
        plot_boarder(
            lcbin.t_phasefold, lcbin.fmed, 'o', 
            mew=0, color='Orange',markersize=3.5)

    if type(lcfit) is not type(None):
        lcfit = lcfit.sort_values('t_phasefold')
        plt.plot(lcfit.t_phasefold, lcfit.f, lw=2, color='RoyalBlue', 
                 alpha=0.7)

    plt.xlim(*xl) # reset the x-limits 
    plt.ylim(*yl) # reset the x-limits 
    AddAnchored(label,prop=tprop,frameon=True,loc=3)
    plt.xlabel('Time Since Transit (days)')

@print_traceback
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
    plot_boarder(
        lcbin[xdata], lcbin.fmed,'o', markersize=4, mew=0, color='Orange'
    )

    xl = (-3.0 * pipe.fit_tdur, 3.0 * pipe.fit_tdur )
    fmed = lcbin[lcbin[xdata].between(*xl)].fmed
    yl = roblims(fmed, 5 , 0.5)
    plt.xlim(*xl)
    plt.ylim(*yl)

    label = "ph = {:.2f}".format(phase_shift)
    AddAnchored(label,prop=tprop,frameon=True,loc=3)
    plt.xlabel('Time Since Reference Phase (days)')
    return lc, lcbin

@print_traceback
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
        lc_single = lc_single.sort_values('phase')
        plt.plot(lc_single.phase,lc_single.dM + yshift,ms=4)
        yshift -= ystep

    plt.xlim(-0.25,0.75)
    plt.xlabel('Phase')
    plt.ylabel('Transit Depth (Box Width = {} cadences)'.format(twd) )
    plt.axvline(0, alpha=0.1, lw=10, color='m', zorder=1)
    plt.axvline(0.5, alpha=0.1, lw=10, color='m', zorder=1)
    AddAnchored("SES Stack", prop=tprop, frameon=True, loc=2)

@print_traceback
def transit_stack(pipe, ystep=None):
    if ystep==None:
        depth = pipe.fit_rp**2
        ystep = depth*2

    lcdt = pipe.lcdt
    lcfit = pipe.lcfit.sort_values('t_phasefold')
    g = pipe.lcdt.groupby('transit_id')

    yshift = 0
    for transit_id, idx in g.groups.iteritems():
        lc = pipe.lcdt.ix[idx]
        color = ['RoyalBlue','Tomato'][transit_id % 2]

        x,y = lc.t_phasefold,lc.f + yshift
        #lines, = plot_boarder(x,y,'o',markersize=4,color=color)
        lines, = plot_boarder(x,y,'.',markersize=6,color='k')
        zorder = lines.get_zorder()-0.1
        l, = plt.plot(x, y, lw=1, color=color, alpha=0.5, zorder=zorder )
        plt.plot(lcfit.t_phasefold, lcfit.f + yshift, '--', lw=1.7)
        yshift -= ystep

    xl = lcdt.t_phasefold.min(), lcdt.t_phasefold.max()
    plt.xlim(*xl)
    plt.xlabel('Time Since Mid Transit (days)')
    AddAnchored("Transit Stack",prop=tprop,frameon=True,loc=2)

@print_traceback
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
    xl = x.min() -1, x.max() + 1 
    xlabel = 'transit-id'

    yspan = y.ptp()
    yl = y.min() - 1 * yspan, y.max() + 1 * yspan


    plt.errorbar(x, y, yerr=yerr, fmt='o', ms=5, mew=0, capsize=0)
    plt.xlim(*xl)
    plt.ylim(*yl)
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

def plot_boarder(*args,**kwargs):
    kwargs_bg = dict(**kwargs)
    kwargs_bg['color'] = 'k'
    kwargs_bg['markersize'] += 1.6
    plt.plot(*args,**kwargs_bg)
    lines = plt.plot(*args,**kwargs)
    return lines
