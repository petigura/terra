"""
My attempt at a python implementation of BLS.
"""
import os
import sys
import numpy as np
from numpy import array
import pdb

import matplotlib.pylab as plt
from keptoy import P2a,a2tdur,ntrans

# Global parameters:
cad = 30./60./24.

# The minimum acceptable points to report s2n
nptsMi =10

# bpt - the number of bins in folded transit.
bpt = 5.

# Filling factor.  Require boxes have at least f of filling factor.
flFact = 2./3

# blsw runs tries millions of P phase combinations.  We want to
# compute the FAP by fitting the PDF and extrapolating to the high S/N
# regime.  nhist is the number of s2n points to return.
nS2nFAP = 1e6 

def blsw(t0,f0,PGrid,retph=False):
    """
    Compute BLS statistic for a the range of periods specified in PGrid.

    Returns:    
    return s2nGrid,s2nFAP,ntrials    
    """
    # Protect the input vectors.
    t = t0.copy()
    f = f0.copy()

    # For a given period, there is an expected transit duration
    # assuming solar type star and equatorial transit.  tmi and tma
    # set how far from this ideal case we will search over.
    ftdur = array([0.75,1.25])
    ng = len(PGrid)
    
    f -= np.mean(f)
    ff = f**2
    
    s2nGrid= np.zeros(ng) - 1
    phGrid= np.zeros(ng)

    if retph:
        phGridl = []
        s2nGridl = []

    s2nFAP = array([])

    tbase = t.ptp()
    ntrials = 0. # Counter for the total number of (P,ph) tests

    # calculate maximum eggress
    npad = ftdur[1] * a2tdur( P2a( max(PGrid) ) ) / cad
    npad *= 2 #
    for i in range(ng):
        # Phase fold the data according to the trial period.        
        P = PGrid[i]
        ph = np.mod(t/P,1.)

        # For this particular period, this is the expected transit
        # duration.
        tdur = a2tdur( P2a(P)  )
        phdur = tdur / P

        # We bin so there are bpt bins in a folded transit.
        nb = np.ceil(1. / phdur * bpt)
        bwd = tdur / bpt
        bins = np.linspace(0,1,nb+1)

        # Calculate the following quantities in each bin.
        # 1. sum of f 
        # 2. sum of f**2 
        # 3. number of points in each bin.
        sb,be = np.histogram(ph,weights=f,bins=bins)
        ssb,be = np.histogram(ph,weights=ff,bins=bins)
        cb,be = np.histogram(ph,bins=bins)

        # How many points do we expect in our transit?
        fb = ntrans(tbase,P,.9)*(bwd/cad)

        # Account for the edge effect.  Pad out arrays by atleast
        # twice the max imum expected transit length
        npad = np.ceil(bpt * 2 * ftdur[1])
        sb  = np.append(sb ,sb[:npad])
        ssb = np.append(ssb,ssb[:npad])
        cb  = np.append(cb ,cb[:npad])
        
        # We compute the sums of sb, ssb, and cb over a trial transit
        # width using a convolution kernel.  We will let that kernel
        # have a small range of widths, which will let us be sensitive
        # to transits of different lengths.

        kwdMi = int( ftdur[0] * bpt ) 
        kwdMa = int( ftdur[1] * bpt ) + 1
        kwdArr = np.arange(kwdMi,kwdMa+1)        
        
        ntdur = kwdMa-kwdMi+1
        s2n = np.empty((nb + npad - kwdMa + 1,ntdur))

        for j in range(ntdur):
            # Construct kernel

            kwd = kwdArr[j]
            kern = np.zeros(kwdMa)
            kern[:kwd] = 1

            # Sum the following quantities in transit
            # 1. f, data values
            # 2. ff, square of the data values
            # 3. n, number of data points.
            st  = np.convolve(sb,kern,mode='valid')
            sst = np.convolve(ssb,kern,mode='valid')
            nt  = np.convolve(cb,kern,mode='valid')

            # Designate a point as a gap if it has less than fill of
            # the expected number of points
            gapid = np.where(nt < flFact*fb*kwd)[0]

            # Average depth
            df    = st / nt

            # Standard deviation of the points in tranist
            sigma = np.sqrt( sst / nt - df**2 )
            s2n[::,j] = -df / (sigma / np.sqrt(nt) ) 
            s2n[gapid,j] = 0 

        # Compute the maximum over trial phases
        s2nFlat = s2n.max(axis=1)
        idMa = np.nanargmax(s2nFlat)

        s2nGrid[i] = s2nFlat[idMa]
        phGrid[i]  = idMa / nb

        if retph:
            phGridl.append(np.linspace(0 , len(df)/nb , len(df) ))
            s2nGridl.append(s2nFlat)

    if retph:
        return phGridl,s2nGridl
    else:
        return s2nGrid

def grid(tbase,ftdurmi,Pmin=100.,Pmax=None,Psmp=0.5):
    """
    Make a grid in (P,ph) for BLS to search over.

    ftdurmi - Minimum fraction of tdur that we'll look for.

    Pmin - minumum period in grid 

    Pmax - Maximum period.  Defaults to tbase/2, the maximum period
           that has a possibility of having 3 transits

    phsmp - How finely to sample phase (in units of minimum tdur
            allowed)
    
    Psmp - How finely to sample period?  The last transit of the
           latest trial period of neighboring periods must only be off by a
           fraction of a transit duration.
    """

    if Pmax == None:
        Pmax = tbase/2.

    P,ph = array([]),array([])
    P0 = Pmin

    while P0 < Pmax:
        
        tdur = a2tdur( P2a(P0)  ) # Expected transit time.
        tdurmi = ftdurmi * tdur

        P = np.append( P, P0 )

        # Threshold is tighter because of accordian effect.
        dP = Psmp * tdurmi #/ (nt -1)

        P0 += dP
        
    return P

def fap(t0nl, f0nl):
    """

    """


    
    nb = 20
    
    # Histogram the values between s2n 3 and 5
    hist,bin_edges = histogram(s2nFAP,range=[0,20],bins=nb)
    
    x = (bin_edges[:-1]+bin_edges[1:])/2
    id = where((x > 3) & (x < 5))[0]

    if len(where(hist[id] < 10 )[0] ) != 0:
        sys.stderr.write('Not enough points to fit PDF\n')
        print x[id],hist[id]
        return None

    p = polyfit(x[id],log10(hist[id]),1)
    s2nfit = polyval(p,s2n)
    s2nfit = 10**s2nfit

    return s2nfit
