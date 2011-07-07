from matplotlib.pylab import plt
import numpy as np

from numpy import *

import keptoy
import pbls
import blsw
import find_blocks

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

