"""
My attempt at a python implementation of BLS.
"""

import os
import numpy as np
from numpy import *

from scipy import weave
from scipy.weave import converters

from scipy.ndimage import median_filter
from scikits.statsmodels.tools.tools import ECDF

import atpy

from peakdetect import peakdet
from keptoy import P2a,a2tdur
import detrend
import medfilt

# Opens file outside the loop
fid = open('pycode/phloop.c') 
code = fid.read()
fid.close()

def blsw(t,f,grid,g0,g1):
    # For a given period, there is an expected transit duration
    # assuming solar type star and equatorial transit.  tmi and tma
    # set how far from this ideal case we will search over.
    ftdur = array([0.5,1.25])
    ng = len(g0)

    # For an assumed transit duration, set the number of bins in
    # transit.  The number should be > 1, so that the intransit points
    # aren't polluted by out of transit points.


    tstop = t[-1]

    f -= np.mean(f)

    Puniq = unique(grid[0])
    ngrid = len(grid[0])
    s2ntot = zeros(ngrid).astype(float) - 1 
    ovarrtot = zeros(ngrid).astype(float) - 1 


    for P in Puniq:
        idPuniq = where( P == grid[0] )[0]
        pharr = grid[1][idPuniq]
        nph = len(pharr)


        # Phase fold the data according to the trial period.        
        phData = mod(t/P,1.)

        # Sort the data according to increasing phase.
        sData = argsort(phData)
        fData = f[sData]
        phData = phData[sData]



        tdur = a2tdur( P2a(P)  ) # Expected transit time
        phdur = ftdur*tdur / P # Allowed transit times in units of phase


        # Zero-out working arrays
        s2narr = zeros(nph).astype(float) - 1 
        ovarr =  zeros(nph).astype(float) - 1 

        # EEBLS phase and flux arrays so we can wrap around.
        # Padd out to the first phase point that is greater than the
        # longest expected transit

        idPad = where( phData > phdur[1] )[0][0]

        phData = append(phData,phData[0:idPad])
        fData  = append(fData,fData[0:idPad])

        phData = phData.astype(float)

        tstop=float(tstop) # Weave barfs unless these are floats
        P=float(P)
        weave.inline(code,
                     ['t','g0','g1','ng','tstop',
                      'P','pharr','nph','phData','fData','phdur',
                      'ovarr','s2narr'],
                     type_converters=converters.blitz,
                     verbose=2
                     )
        
        s2ntot[idPuniq]   = s2narr
        ovarrtot[idPuniq] = ovarr

    return s2ntot,ovarrtot

def blsw_loop2(y,ibi,kma,kmi,kkmi):
    """
    This is the inner two nested for loops in the BLS algorithm

    Given: 
    y - an array of binned values
    ibi - an array of the number of points in each bin.
    
    Return:
    power - maximum signal residue for that period
    jn1   - index of optimal ingress
    jn2   - index of optimal egress
    rn3   - number of unbinned points in the optimal transit
    s     - sum of the points in the window

    Example:
    y   = [0,0,0,1,2,0,0,0,0,0]
    ibi = [1,1,1,1,2,1,1,1,1,1]
    should return
    jn1 = 3
    jn2 = 4
    s   = 3
    pow = 0.375
    """

    nb = len(y)
    rn = float(np.sum(ibi))
    res = weave.inline(code,['y','ibi','kma','kmi','kkmi','rn','nb'],
                       type_converters=converters.blitz)
    return res

def blswrap(t,f,nf=200,fmin=None,fmax=1.):
    """
    blswrap sets the input arrays for the BLS algorithm.

    Description of the inputs:
    t       : array
              time
    f       : array
              data values
    blsfunc : function
              must have the following signature
              blsfunc(t,f,nf,fmin,df,nb,qmi,qma,n)
              See bls for description of each input
    """

    tot = t[-1] - t[0]

    if fmin == None:
        # Twice the lowest fft freq
        fmin = 2./tot

    # We require there be one dip.
    # TODO: should this be 2/T?
    if fmin < 1./tot:
        print ' fmin < 1/T !!'
        return None

    farr = np.logspace( np.log10(fmin), np.log10(fmax), nf)
    p,ph,qtran,df = blsw(t,f,farr)

    parr = 1/farr

    out = {'p'    :p   ,
           'farr' :farr,
           'parr' :parr,
           'ph':ph,
           'df':df,
           'qtran':qtran,

           # Convience
           'tdur':qtran*parr,
           'f':f,
           't':t,
           }

    return out
import sys

def grid(tbase,ftdurmi,Pmin=100.,Pmax=None,phsmp=0.5,Psmp=0.5):
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


    Note:  Takes ~ 2 min to run with the default settings (tbase = 1000).
           Could speed this up in C.

           7 M entries.
    """

    if Pmax == None:
        Pmax = tbase/2.

    P,ph = array([]),array([])

    P0 = Pmin
    while P0 < Pmax:
        
        tdur = a2tdur( P2a(P0)  ) # Expected transit time.
        tdurmi = ftdurmi * tdur
        dt = phsmp*tdurmi   # Time corresponding to phase step
        nph = ceil(P0/dt)
        
        pharr = linspace(0,1,nph)
        
        P = append( P , zeros(nph)+P0 )
        ph = append( ph , pharr )


        # Calculate new Period
        # Maximum number of transit.  True for when ph = 0 
        nt = floor(tbase/P0)

        # Threshold is tighter because of accordian effect.
        dP = Psmp * tdurmi #/ (nt -1)

        P0 += dP
        
    return P,ph






def plotspec(grid,s2n):

    Puniq = unique(grid[0])
    nPuniq = len(Puniq)
    s = zeros(nPuniq) -1 
    for i in range(nPuniq):
        idPuniq = where( grid[0] == Puniq[i] )[0]
        s[i] = max(s2n[idPuniq])


    return Puniq,s

def maskgap(t,f,g0,g1):
    """

    """
    t = ma.masked_array(t,mask=False)
    f = ma.masked_array(f,mask=False)
    for i in range(len(g0)):
        t = ma.masked_inside(t,g0[i],g1[i])

    f.mask= t.mask
    f = f.compressed()
    t = t.compressed()
    return t,f
