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
import sys
from scikits.statsmodels.tools.tools import ECDF

import blsw
from keptoy import lightcurve

# Don't believe any faps that are below this value.  The extrapolation
# of the CDF is surely breaking down.
fapfloor = 1e-7
pickdir = 'pickles/run3/'

s2n = [4,5,6,8,10]
files = [open('pickle/run3/tb-1000_s2n-%d_1000.pickle' % s,'rb') for s in s2n]
ns2n = len(s2n)

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
    f = open('pickle/null1000.pickle')
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
            o = peakfill(o)

            miper,mifap = peakfap( nper, npow, o['parr'], o['p'] )
            mapow = o['p'][ where( o['parr'] == miper ) ] 
            iP = d['P']
            if plotsave != None:
                perlim = [ nper.min() , nper.max() ]
                perg,pwg = mgrid[ perlim[0]+1 : perlim[1]-1: 10j,
                                  1e-6 : 5e-6 : 100j]
                fapg = nullgrid(nper,npow,perg,pwg)
                plotnull(perg,pwg,log10(fapg) )
                ax = plt.gca()
                ax.plot( o['parr'] ,o['p'],'k')
                if abs(iP - miper)/iP > 0.05:
                    ax.plot( [miper],[mapow],'or')
                else:
                    ax.plot( [miper],[mapow],'og')

                fig = plt.gcf()
                figt = 'FAP %.2e \n Seed %d' % (mifap,d['seed']) 
                
                fig.text(0.9,0.8,figt, 
                         ha="center",
                         bbox = dict(boxstyle="round", fc="w", ec="k") )

                fig.savefig('frames/%s_%03d.png' % (plotsave,d['seed']) )
                
            res.append( {'mifap':mifap,'miper':miper} )

        except Exception, err:
            sys.stderr.write('ERROR: %s\n' % str(err))
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

    mlog = int(log10(fapfloor))

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
    mifapid = abs(fap).argmin()
    mifap = fap[mifapid]

    if mifap < fapfloor:
        bperid = pow.argmax()
    else:
        bperid = mifapid
    
    # Account for the fact that a peak could have been anywhere.  
    # This assumes that all the trial periods are independent which is
    # Not true.
    nind = len(per)

    mifap = fap[bperid] * nind
    miper = per[bperid]

    return miper,mifap

def peakfill(o):
    upsmp = 10.
    thresh = 3e-6
    pkid = where(o['p'] > thresh)[0]
    
    sid = array([])
    for p in pkid:
        sid = append(sid,p-1)
        sid = append(sid,p)
        sid = append(sid,p+1)

    sid = unique(sid)
    
    ffn = array([])
    for i in range( 0, len(sid) -1 ):
        fcrs = o['farr']
        ffl = linspace(fcrs[ sid[i] ], fcrs[ sid[i+1] ], upsmp)
        ffn  = append( ffn, ffl)

    p,ph,qtran,df = blsw.blsw(o['t'],o['f'],ffn)

    o['p']     = append(o['p']     ,p     )
    o['farr']  = append(o['farr']  ,ffn   )
    o['parr']  = append(o['parr']  ,1/ffn )
    o['ph']    = append(o['ph']    ,ph    )
    o['df']    = append(o['df']    ,df    )
    o['qtran'] = append(o['qtran'] ,qtran )

    return o

def fap_s2n():
    """
    Show how the FAP changes as a function of s2n    
    """
    
    fig = plt.gcf() 

    for i in range(ns2n):
        s = s2n[i]
        file = files[i]

        out = pickle.load(file)
        res,darr = out['res'],out['darr']

        fap = array([r['mifap'] for r in res])
        oP = array([r['miper'] for r in res])
        iP = array([r['P'] for r in darr])
        

        fail =  abs(oP-iP)/iP > 0.1
        y = log10(fap)

        ax = fig.add_subplot(ns2n,1,i+1)
        bins = linspace(-15,5,41)
        
        ax.hist( y[where(~fail)] ,color='g',bins=bins,label='Good')
        ax.hist( y[where(fail)] ,color='r',alpha=0.8,bins=bins,label='Fail')

        plt.legend(title='S/N - %d' %  round(darr[0]['s2n']) , loc='best')

    ax.set_xlabel('log(FAP)')
    plt.show()
        
def errors():

    print """
Null Hypothesis:  There is no planet with period P = oP.
|--------------------------------------------|
|                      P != oP      P = oP   |
|                      -------      -------  |
|Reject null           False Pos    True Pos |
|Fail to reject null   True Neg     False Neg|
|--------------------------------------------|

S/N   False Pos  True Pos  True Neg  False Neg
---   ---------  --------  --------  ---------""" 

    for i in range(ns2n):
        s = s2n[i]

        file = files[i]

        out = pickle.load(file)
        res,darr = out['res'],out['darr']

        fap = array([r['mifap'] for r in res])
        oP = array([r['miper'] for r in res])
        iP = array([r['P'] for r in darr])
        nsim = 1.0*len(fap)
        
        # Good fap
        gfap = fap < 1e-2
        
        # Wrong period
        wper = abs(oP-iP)/iP > 0.1

        fp = len( where( gfap &  wper )[0] )  # False Positive
        tp = len( where( gfap &  ~wper )[0] ) # True positive
        tn = len( where( ~gfap & wper )[0] ) # True positive
        fn = len( where( ~gfap & ~wper )[0] ) # True positive
        print """ %02d   %.3f      %.3f     %.3f     %.3f """ %\
            (s,fp/nsim,tp/nsim,tn/nsim,fn/nsim) 

def iPoP():
    """
    Plot input period versus output period.
    """

    fig = plt.gcf() 

    for i in range(ns2n):
        s = s2n[i]
        file = files[i]

        out = pickle.load(file)
        res,darr = out['res'],out['darr']

        fap = array([r['mifap'] for r in res])
        oP = array([r['miper'] for r in res])
        iP = array([r['P'] for r in darr])
        
        fail =  abs(oP-iP)/iP > 0.1
        nfail = len(where(fail)[0])
        y = log10(fap)

        ax = fig.add_subplot(ns2n,1,i+1)

        ax.plot( iP, oP, '.b',ms=3)
        ax.set_xlabel('Input Period')
        ax.set_ylabel('Best Period')

        tt = """
S/N  - %d
fail - %d""" % (round(s),nfail)

        harm = [1/2.,2.]

        for h in harm:
            ax.plot( iP, h*iP, 'r',alpha=0.5)

        ax.plot( iP, iP, 'g',alpha=0.5)

        left = ax.get_xlim()[0]
        top = ax.get_ylim()[1]

        ax.text(left,top,tt,bbox=dict(facecolor='white'),fontsize='large',va='top',ha='left')





    plt.show()
