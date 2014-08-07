from matplotlib.pylab import plt
import numpy as np

from numpy import *

import keptoy
import pbls
import blsw
import find_blocks
import bb

from scipy.ndimage import median_filter
from scipy.stats import ks_2samp


from peakdetect import peakdet

def p2d(p,farr,ph):
    """
    Shows the power for the BLS 2D spectrum
    """

    # first index is plotted as y,this should be the other way around
    

    p = np.swapaxes(p,0,1)
    f = plt.gcf()
    f.clf()

    # create two axes, one for the image, the other for the profile.
    r1d = [0.1,0.75,0.8,0.2]
    r2d = [0.1,0.1,0.8,0.65]

    ax1d = plt.axes(r1d)
    ax2d = plt.axes(r2d,sharex=ax1d)

    power = np.max(p,axis=0)
    ax1d.plot(farr,power)
    ax1d.set_ylabel('Power') 


    ax2d.pcolorfast(farr,ph,p)
    ax2d.set_xlabel('frequency days^-1') 
    ax2d.set_ylabel('phase of trans / 2 pi') 


    plt.show()


def phasemov():

    parr=np.linspace(0,2*pi,30)
    for i in range(len(parr)):
        f,t = keptoy.lightcurve(s2n=1000,P=10.,phase=parr[i])

        nf,fmin,df,nb,qmi,qma,n = pbls.blsinit(t,f,nf=1000)
        p,farr,ph = blsw.blswph(t,f,nf,fmin,df,nb,qmi,qma,n)
        p2d(p,farr,ph)

        f = plt.gcf()
        f.text(0.8,0.7, "EEBLS Phase %.2f" % (parr[i]) ,ha="center",
               bbox=dict(boxstyle="round", fc="w", ec="k"))
        f.savefig('frames/ph%02d.png' % i)
        
        plt.show()


def blocks(t,f,last,val):
    """
    Plot lines for the Bayesian Blocks Algorithm
    """

    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111)

    ax.plot(t,f,'o',ms=1,alpha=0.5)

    cp = unique(last)
    n = len(t)
    idxlo = cp                    # index of left side of region
    idxhi = append(cp[1:],n)-1 # index of right side of region 

    ax.hlines(val[idxhi],t[idxlo],t[idxhi],'red',lw=5)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Flux (normalized)')

    plt.show()


def ncp(s2n):
    """
    Explore the output of BB as we change the prior on the number of
    change points
    """
    
    nncp = logspace(0.5,1.5,9)
    
    f,t = keptoy.lightcurve(s2n=s2n)
    sig = zeros(len(t)) + std(f)

    for i in range( len(nncp) ):
        last,val = find_blocks.pt( t,f,sig,ncp=nncp[i] )
        blocks(t,f,last,val)
        
        ax = plt.gca()
        ax.set_title('s2n - %.1e, cpts prior - %.2e' % (s2n,nncp[i]) )
        fig = plt.gcf()

        fig.savefig('frames/s2n-%.1e_%02d.png' % (s2n,i) )
        plt.show()



def phasemov():
    ph=linspace(0,2*pi,30)
    for i in range(len(ph)):
        f,t = keptoy.lightcurve(s2n=1000,P=10.,phase=ph[i])

        nf,fmin,df,nb,qmi,qma,n = pbls.blsinit(t,f,nf=1000)
        p = blswph(t,f,nf,fmin,df,nb,qmi,qma,n)
        f = plt.gcf()
        f.clf()
        ax = f.add_subplot(111)
        ax.imshow(p,aspect='auto')

        ax.set_xlabel('bin number of start of transit')
        ax.set_ylabel('frequency')
        ax.set_title('2D BLS spectrum - phase %.2f' % ph[i])
        f.savefig('frames/ph%02d.png' % i)

        plt.show()


def blsbb():
    """
    Simulate a time series.  Compare two bls spectra.

    1. Normal BLS
    2. Blocks returned by BB

    Also plot the timeseries.

    """


    s2n = logspace(0.5,2,8)

    for i in range( len(s2n) ):

        f,t = keptoy.lightcurve(s2n = s2n[i],tbase=600)

        sig = zeros(len(t)) + std(f)
        last,val =  find_blocks.pt(t,f,sig,ncp=5)
        fb = bb.get_blocks(f,last,val)

        o = pbls.blswrap(t,f,blsfunc=blsw.blsw,nf=1000)
        ob = pbls.blswrap(t,fb,blsfunc=blsw.blsw,nf=1000)

        fig = plt.gcf()
        fig.clf()


        ax = fig.add_subplot(211)


        ax.plot(t,f+1,'o',ms=1,alpha=0.5)
        
        cp = unique(last)
        n = len(t)
        idxlo = cp                    # index of left side of region
        idxhi = append(cp[1:],n)-1 # index of right side of region         
        ax.hlines(val[idxhi],t[idxlo],t[idxhi],'red',lw=5)
        ax.set_ylabel('Flux (normalized)')
        ax.set_xlabel('Flux (normalized)')



        ax = fig.add_subplot(212)
        ax.plot(o['farr'],o['p']    ,label = 'BLS')
        ax.plot(ob['farr'],ob['p']  ,label = 'BLS + BB')
        ax.set_xlabel('Frequency days^-1')
        ax.set_ylabel('SR')
        ax.legend()

        fig.text(0.9,0.9, "S/N - %.1e" % (s2n[i]) ,
                 ha="center",fontsize=36,
                 bbox=dict(boxstyle="round", fc="w", ec="k"))

        fig.savefig('frames/blsbb%02d.png' % i)
        plt.show()



def bbfold(P):
    

    # Trial Data:
    tbase = 90.
    f,t = keptoy.lightcurve(tbase=tbase,s2n=5,P=10.1)
    
    o = pbls.blswrap(t,f,blsfunc=blsw.blsw)
    farr = linspace(min(o['farr']),max(o['farr']),100)
    sig = zeros(len(f)) + std(f)

 #   for fq in farr:

    tfold = mod(t,P) # Fold according to trial period.
    sidx = argsort(tfold)
    tfold = tfold[sidx]
    ffold = f[sidx]

    last,val = find_blocks.pt(tfold,ffold,sig,ncp=8)

    fig = plt.gcf()
    fig.clf()

    ax = fig.add_subplot(111)
    ax.plot(tfold,ffold,'o',ms=1,alpha=0.5)

    cp = unique(last)
    n = len(t)
    idxlo = cp                    # index of left side of region
    idxhi = append(cp[1:],n)-1 # index of right side of region         
    ax.hlines(val[idxhi],tfold[idxlo],tfold[idxhi],'red',lw=5)
    ax.set_ylabel('Flux (normalized)')

    plt.show()


def perfind():
    """
    Test how good the KS test is at descriminating a real planet from
    a none.    
    """

    P = 200.
    tbase = 1000.
    nf = 500

    s2n = logspace( log10(3),log10(15),5 )
    print "S/N sig-noise noise-noise" 
    for s in s2n:
        # Generate a lightcurve.
        f,t = keptoy.lightcurve(P=P,tbase=tbase,s2n=s)
        o = blsw.blswrap(t,f,nf=nf,fmax=1/50.)

        # Generate a null lightcurve
        fn = std(f)*random.randn(len(f))
        on = blsw.blswrap(t,fn,nf=nf,fmax=1/50.)

        # Generate another null lightcurve
        fn2 = std(f)*random.randn(len(f))
        on2 = blsw.blswrap(t,fn2,nf=nf,fmax=1/50.)

        o['p'] -= median_filter(o['p'],size=200)
        on['p'] -= median_filter(on['p'],size=200)
        on2['p'] -= median_filter(on2['p'],size=200)
        
        print "%.2f %.2e %.2e %.2f" % \
            (s,(ks_2samp(o['p'],on['p']))[1],(ks_2samp(o['p'],on2['p']))[1],
             o['parr'][argmax(o['p'])])
        

        fig = plt.gcf()
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(o['parr'],o['p'],label='Signal')
        ax.plot(o['parr'],on['p'],label='Noise')        
        ax.plot(o['parr'],on2['p'],label='Noise2')
        ax.legend()
        plt.show()


def perfind2():
    
    P = 200.
    tbase = 1000.
    nf = 5000


    s2n = logspace( log10(3),log10(15),5 )
    print "S/N sig-noise noise-noise" 
    for s in s2n:
        # Generate a lightcurve.
        f,t = keptoy.lightcurve(P=P,tbase=tbase,s2n=s)
        o = blsw.blswrap(t,f,nf=nf,fmax=1/50.)

        # Subtract off the trend
        o['p'] -= median_filter(o['p'],size=200)

        # Find the highest peak.
        mxid = argmax(o['p'])
        mxpk = o['p'][mxid]

        # Find all the peaks        
        mxt,mnt = peakdet(o['p'],delta=1e-3*mxpk,x=o['parr'])
        mxt = array(mxt)

        # Restrict attention to the highest 100 but leaving off top 10
        t1id = where( (mxt[::,1] > sort( mxt[::,1] )[-100]) &
                      (mxt[::,1] < sort( mxt[::,1] )[-10]) )



        fig = plt.gcf()
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(o['parr'],o['p'],label='Signal')
        ax.scatter( mxt[t1id,0],mxt[t1id,1],label='Tier 1' )
        
        # tpyical values of the highest 100 peaks.
        hair = median(mxt[t1id,1]) 


        left  = min(o['parr'])
        right  = max(o['parr'])
        ax.hlines(hair,left,right)

        if mxpk > 3*hair:
            print "Peroid is %.2f" % (o['parr'][mxid])
        else:
            print "No signal"
        
        ax.legend()
        plt.show()

