"""
Motivation:

This module contains functions for determining the `efficacy' of
transit detection algorithms.  

Goals:
- Manage the creation of monte carlo data.
- Profiling different algorithms in a systematic way.
- Storing the results.

Requirements:
- Functions should should take only t,f as required arguments

What is Efficacy?
-----------------
- The metric for the efficacy of a function will be the number of
correctly identified transit signatures over the number of mock data
sets.
- Function must identify the correct trial period to within 1%
- Function must determine the phase of the transit to 1%
"""


import itertools
import cPickle as pickle
import matplotlib.pylab as plt
from numpy import *
from numpy.random import random

from scikits.statsmodels.tools.tools import ECDF

import blsw
from keptoy import lightcurve

def init(tbase=[30,600]   ,ntbase = 30,
         s2n = [100.,100.], ns2n = 1,
         ntr=20, null=False):

    per = 200.
    wphase = 2*pi      # randomize phase over entire cycle
    wper   = per / 10. # dither period by 10%

    # Initialize test parameters    
    s2n   = log10( array(s2n) )
    tbase = log10( array(tbase))

    s2n = logspace(s2n[0],s2n[1],ns2n)
    tbase = logspace(tbase[0],tbase[1],ntbase)
    par = itertools.product(*[s2n,tbase])

    darr = []
    seed = 0

    for p in par:
        for i in range(ntr):
            
            phase = wphase*random()
            dper   = wper*random()
            d = {'s2n':p[0],'tbase':p[1],'P':per+dper,'phase':phase,
                 'null':null,'seed':seed}

            darr.append(d) 
            seed += 1
        
    return darr

def profile(darr,save=None,plotsave=None):
    """
    Will profile a transit search algorithm over a range of:
    - S/N
    - tbase
    
    Arguments
    func - a function the finds the period.  Must only accept f,t
    time series 
    darr - an array of dictionary arguments that will be passed to
    `lightcurve`  These are the parameters for the fits. Must include:
     P     - period
     """


    f = open('null1000.pickle')
    n = pickle.load(f)

    nper = n['parr']
    npow = array([p for p in n['p']])
    count = 0

    res = []
    for d in darr:
        count += 1
        f,t = lightcurve(**d) 

        try:
            o = blsw.blswrap(t,f,nf=5000,fmax = 1/50.)
            miper,mifap = peakfap( nper, npow, o['parr'], o['p'] )
            mapow = o['p'][ where( o['parr'] == miper ) ] 

            if plotsave != None:
                perlim = [ nper.min() , nper.max() ]
                perg,pwg = mgrid[ perlim[0]+1 : perlim[1]-1: 10j,
                                  1e-6 : 5e-6 : 100j]
                fapg = nullgrid(nper,npow,perg,pwg)
                plotnull(perg,pwg,log10(fapg) )


                ax = plt.gca()
                ax.plot( o['parr'] , o['p'] )
                ax.plot( [miper],[mapow],'or')

                fig = plt.gcf()

                figt = """
FAP %.2e \n
Seed %d 
""" % (mifap,d['seed']) 

                fig.text(0.9,0.8,figt, 
                         ha="center",
                         bbox = dict(boxstyle="round", fc="w", ec="k") )
                fig.savefig('frames/%s_%03d.png' % (plotsave,d['seed']) )

            res.append( {'mifap':mifap,'miper':miper} )
            
        except ValueError:
            print "Value Error"
            res.append( {'mifap':0,'miper':0} )

    if save != None:
        f = open(save,'wb')
        pickle.dump({'res':res,'darr':darr},
                    f,protocol=2)

    return res

def pprofile(darr,save=None,plotsave=''):
    """
    Multicore version of previous code
    """

    from IPython.kernel import client
    mec = client.get_multiengine_client()

    id = mec.get_ids()
    mec.scatter('darr',darr)
    mec.scatter('id',id)
    mec.push({'plotsave':plotsave})

    mec.execute(r"""
from qalg import profile; 
res = profile(darr,plotsave='%s' % (plotsave)  ) ;
"""
)
    res = mec.gather('res')

    if save != None:
        f = open(save,'wb')
        pickle.dump(
            {'res':res,'darr':darr},
            f,protocol=2)
        f.close()

    return res


def pnull(darr,func,save=None):
    """
    Multicore version of previous code
    """

    from IPython.kernel import client
    mec = client.get_multiengine_client()

    mec.scatter('darr',darr)
    mod = func.__module__
    name = func.__name__

    mec.execute("""
from qalg import profile; 
from %s import %s ;
res = profile(darr,%s);
""" % (mod,name,name)
)
    res = mec.gather('res')

    p = [r['p'] for r in res]
    parr = res[0]['parr']

    res = {'p':p,'parr':parr}
    if save != None:
        f = open(save,'wb')
        pickle.dump(res,f,protocol=2)
        f.close()

    return res


def nullgrid(per,pw,perg,pwg):
    """
    per - array of bls periods
    pw  - array of MC powers
    perg - periods of points we're interested in sampling
    pwg - power points we're interested in sampling
    """
    
    # bin the period array into this many points.
    nb = 10 
    perbin = linspace( per.min(), per.max(), nb + 1 )
    
    gshp = perg.shape

    perg = perg.flatten()    
    pw = pw.flatten()
    pwg = pwg.flatten()

    pergbid = digitize( perg,perbin )
    perbid = digitize( per,perbin )
    
    fapg = zeros( perg.shape ) - 1 
    for i in range(1,nb+1):

        id_cb  = where( perbid == i )
        gid_cb = where( pergbid == i )

        # Avoid empty bins
        if gid_cb[0].size != 0:
            # gather up all the MC runs from that period bin.
            pw_cb  = pw[ id_cb]
            pwg_cb = pwg[ gid_cb ]

            cdf = ECDF(pw_cb.flatten())
            fapg[ gid_cb ] = fap(cdf,pwg_cb)

    fapg = reshape(fapg,gshp)
    
    return fapg

def plotnull(x,y,z):

    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111)

    mlog = floor(sort(unique(z))[1])

    levelsf = linspace(mlog,0,32)
#    cf = ax.contourf(x,y,z,levelsf,cmap=plt.cm.bone_r)

    # Overplot some lines
    levelsl = linspace(mlog,-1,abs(mlog))
    c = ax.contour(x,y,z,levels=levelsl,colors='green')    
    plt.clabel(c,inline=1,fmt='%i',inline_spacing=10)

    ax.set_xlabel('Period days^-1')
    ax.set_ylabel('Signal Residue')


def fap(cdf,x):
    """
    Given a CDF and values, return the probability, extrapolating in the case
    of low statistics.
    """

    # so this will work with single values.
    x = array(x)

    # Fit a line to the last values.

    fap_data = log10( 1 - cdf.y )[-100:-10]
    xx  = cdf.x[-100:-10]
    maxx = cdf.x.max()

    p = polyfit(xx,fap_data,1)

    fap = zeros(len(x)) - 1
    interpid = where( x <= maxx )
    extrapid = where( x >  maxx )

    fap[interpid] = log10( 1 - cdf( x[interpid] ) )
    fap[extrapid] = polyval(p, x[extrapid] )

    fap = 10**fap
    return fap

def peakfap(nper,npow,per,pow):
    """    
    nper - period array for null run
    npow - power array for null run
    per  - period array for lightcurve spectrum
    pow  - power array for lightcurve spectrum
    """

    fap = nullgrid(nper,npow,per,pow)
    
    # Find the minimum FAP
    id = abs(fap).argmin()
    
    # Account for the fact that a peak could have been anywhere.
    nind = len(per)
    mifap = fap[id] * nind
    miper = per[id]

    return miper,mifap


def eta(xarr,parr):
    x = unique(xarr)
    eta = zeros( len(x) ) 

    for i in range( len(x) ) :
        # How many runs where there at this value?
        ntr = len ( where(xarr == x[i] ) [0] )
        npass = len( where( (xarr == x[i]) & (parr == True) )[0] )
        eta[i] = 1.*npass/ntr

    return x,eta 
