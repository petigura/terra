"""
A module containing python implimentations of BLS.
"""

import numpy as np
from numpy import *
def blscc(t,x,nf,fmin,df,nb,qmi,qma,n):
    """
    Python BLS - a carbon copy of the fortran version

    Fits five parameters:
       - P_0 - Fundemental period
       - q   - fractional time of period in transit
       - L   - low value
       - H   - high value
       - t_0 - epoch of transit       

    Description of the inputs:
    t    : array
           time
    f    : array
           data values
    nf   : int
           number of frequency points to sample.
    fmin : double
           The minimum frequency at which to compute peroidogram
           MUST be greater than 1/ Time Baseline
           default - nyquist
    nb   : number of bins in folded time series at any test point
    qmi  : minimum fractional transit length.  1 year orbit 5e-4
    qma  : maximum fractional transit length.  1 day orbit  2e-2

    The output is called signal residue


    """
#==========================
#   Initialize Variables
#==========================

    minbin = 5
    nbmax = 2000

    u = zeros(n)
    v = zeros(n)

    y  = zeros(nbmax)
    ibi = zeros(nbmax)
    p = zeros(nf)

    # Number of bins specified by the user cannot exceed hard-coded amount
    if nb > nbmax:
        print ' NB > NBMAX !!'
        return None

    # tot is the time baseline
    tot = t[-1] - t[0]
    # The minimum frequency must be greater than 1/T
    # We require there be one dip.
    # To nail down the period, wouldn't we want 2 dips?

    if fmin < 1./tot:
        print ' fmin < 1/T !!'
        return None

#  turn n into a float
    rn=float(n)

# integer ceiling of the product
# bin number of binned points in transit
    kmi = int(qmi*float(nb))

    if kmi < 1:
        kmi=1

#   maximum number of binned points in transit
    kma = int(qma*float(nb)) + 1
#   minimum number of (unbinned) points in transit
    kkmi = int(rn*qmi)
    if kkmi < minbin:
        kkmi = minbin

    bpow=0.

#=================================
#     Set temporal time series
#=================================

    s=0.

#     Subtract start time and mean
    for i in range(n):
        u[i] = t[i]-t[0]
        s=s+x[i]

    s=s/rn # avg value

    # subtract off the mean
    for i in range(n):
        v[i] = x[i]-s

#==============================
#     Triple-nested loop over
#     1. Period
#     2. Phase of transit
#     3. Transit Duration
#==============================


# jf - index for stepping through trial frequencies
    for jf in range(nf):
        f0=fmin+df*float(jf) 
        p0=1./f0 # p0 - period corresponding to the trial frequency

#======================================================
#     Compute folded time series with  *p0*  period
#======================================================
#     Zero-out working arrays
        for j in range(nb):
            y[j]   = 0.
            ibi[j] = 0

#       Phase fold the data according to the trial period
        for i in range(n):
              
#           ph is the "phase" of the ith measurement 
#           (note their is a 2pi missing)
            ph = u[i]*f0

#           modulo(phase,period)
            ph = ph-int(ph)

#           j what bin is the ith data point in?
            j  = int(nb*ph)

#           ibi counts the number of points in the bin
            ibi[j] = ibi[j] + 1

#           y sums the points 
            y[j] = y[j] + v[i]
            
        power=0.

#       Loop over the phase of transit (i - sum over bins)
        for i in range(nb):
            s     = 0.
            k     = 0
            kk    = 0
            nb2   = i+kma
            if nb2 > nb:
                nb2=nb

#           Loop over the duration of transit
            for j in linspace(i,nb2,nb2-i+1).astype(int):
                k = k+1
                kk = kk+ibi[j]
                s = s+y[j]

#               Which (duration,phase) combination that produces peak power?
                if (k > kmi) & (kk > kkmi):
                    rn1 = float(kk)
                    pow = s*s/(rn1*(rn-rn1))

                    if pow > power:
                        power = pow
                        jn1   = i
                        jn2   = j
                        rn3   = rn1
                        s3    = s

        power = sqrt(power)
        p[jf] = power

        if power > bpow:
            bpow  =  power
            in1   =  jn1
            in2   =  jn2
            qtran =  rn3/rn
            depth = -s3*rn/(rn3*(rn-rn3))
            bper  =  p0

    return p,bper,bpow,depth,qtran,in1,in2



def blswrap(t,f,blsfunc=None,nf=200,fmin=None,fmax=1.,nb=1000,qmi=1e-4,qma=1e-1,
            ver=False):
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

    n = len(t)
    tbsln = t.max()-t.min()

    # Check that f is in the right range.
    if fmin == None:
        # Twice the lowest fft freq
        fmin = 2./tbsln

    if fmin < 1./tbsln:
        raise ValueError

    df = (fmax-fmin)/nf

    p,bper,bpow,depth,qtran,in1,in2 = blsfunc(t,f,nf,fmin,df,nb,qmi,qma,n)

    farr = np.linspace(fmin,fmax,nf) 
    parr = 1/farr


    # Calculate phase of mid transit.
    mdt = (1.*in1/nb+qtran/2.)*bper

    if ver:
        print """
peak period     - %.4f
peak power      - %.4f
depth at p      - %.4f
frac trans time - %.4f
first bin       - %i
last bin        - %i
trans mid time  - %.2f
""" % (bper,bpow,depth,qtran,in1,in2,mdt)

    out = {'p'    :p   ,
           'farr' :farr,
           'bper' :bper,
           'bpow' :bper,
           'mdt'  :mdt ,
           'depth':depth,
           'qtran':qtran,

           # Convience
           'phase':2.*pi*in1/nb,
           'tdur':qtran*bper
           }

    return out
