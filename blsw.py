"""
My attempt at a python implementation of BLS.
"""
import numpy as np
from numpy import *

from scipy import weave
from scipy.weave import converters

from keptoy import P2a,a2tdur

def blsw(t,x,nf,fmin,df,nb,qmi,qma,n):
    """
    Python BLS - a modular version of BLS

    Fits five parameters:
       - P_0 - Fundemental period
       - q   - fractional time of period in transit
       - L   - low value
       - H   - high value
       - t_0 - epoch of transit       

    The output is called signal residue
    """
    minbin = 5
    nbmax = 2000

    # For a given period, there is an expected transit duration
    # assuming solar type star and equatorial transit.  tmi and tma
    # set how far from this ideal case we will search over.
    tmi = 0.5
    tma = 2.0

    # For an assumed transit duration, set the number of bins in
    # transit.  The number should be > 1, so that the intransit points
    # aren't polluted by out of transit points.

    ntbin = 10.

    y   = zeros(nbmax)
    ibi = zeros(nbmax)
    p   = zeros(nf)

    # Number of bins specified by the user cannot exceed hard-coded amount
    if nb > nbmax:
        print ' NB > NBMAX !!'
        return None

    # We require there be one dip.
    # TODO: should this be 2/T?
    tot = t[-1] - t[0]
    if fmin < 1./tot:
        print ' fmin < 1/T !!'
        return None

    rn = float(n)


    bpow = 0.
    x -= np.mean(x)

    #=====================================#
    # Triple-nested loop over   index     #
    # 1. Period                 jf        #
    # 2. Phase of transit       i (weave) # 
    # 3. Transit Duration       j (weave) #
    #=====================================#

    farr = np.linspace(fmin,fmin+nf*df,nf) 

    for jf in range(nf):
        f0 = farr[jf]

        P = 1.0/f0
        
        ftdur = a2tdur( P2a(P)  ) / P # Expected fractional transit time

        # Choose nb such that the transit will contain ntbin bins.
        nb = ntbin / ftdur 
        bins = np.linspace(0,1,nb+1)

        qmi = tmi * ftdur
        qma = tma * ftdur

        # kmi is the minimum number of binned points in transit.
        # kma is the maximum number of binned points in transit.
        # kkmi is the minimum number of unbinned points in transit.

        kmi = max( int(qmi*float(nb) ),1 )
        kma = int(qma*float(nb)) + 1
        kkmi = max(int(n*qmi),minbin)

#       Zero-out working arrays
        y   = zeros(nbmax).astype(float)
        ibi = zeros(nbmax).astype(int)

#       Phase fold the data according to the trial period.
        ph = t*f0
        ph = np.mod(ph,1.)

#       Put the data in bins.
        y =   ( np.histogram(ph,bins=bins,weights=x) )[0]
        ibi = ( np.histogram(ph,bins=bins) )[0]

#       EEBLS extend y and ibi so they include kma more points
        y   = np.append(y,y[0:kma])
        ibi = np.append(ibi,ibi[0:kma])

#       Loop over phase and tdur.
        power,jn1,jn2,rn3,s3 = blsw_loop2(y,ibi,kma,kmi,kkmi)

        p[jf] = sqrt(power)

        if power > bpow:
            bpow  =  power
            in1   =  jn1
            in2   =  jn2
            qtran =  rn3/rn
            depth = -s3*rn/(rn3*(rn-rn3))
            bper  =  1./f0

    return p,bper,bpow,depth,qtran,in1,in2

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

    # TODO: opening the file inside the loop is inefficient
    fid = open('ccode/blsw_loop2.c') 
    code = fid.read()
    fid.close()

    res = weave.inline(code,['y','ibi','kma','kmi','kkmi','rn','nb'],
                       type_converters=converters.blitz)
    return res


################################
# OUTPUTTING PHASE INFORMATION #
################################

def blswph(t,x,nf,fmin,df,nb,qmi,qma,n):
    """
    Python BLS - a modular version of BLS

    Fits five parameters:
       - P_0 - Fundemental period
       - q   - fractional time of period in transit
       - L   - low value
       - H   - high value
       - t_0 - epoch of transit       

    The output is called signal residue
    """
    minbin = 5
    nbmax = 2000


    # Number of bins specified by the user cannot exceed hard-coded amount
    if nb > nbmax:
        print ' NB > NBMAX !!'
        return None

    # We require there be one dip.
    # TODO: should this be 2/T?
    tot = t[-1] - t[0]
    if fmin < 1./tot:
        print ' fmin < 1/T !!'
        return None

    rn = float(n)

#   kmi is the minimum number of binned points in transit.
#   kma is the maximum number of binned points in transit.
#   kkmi is the minimum number of unbinned points in transit.

    kmi = max(int(qmi*float(nb)),1)
    kma = int(qma*float(nb)) + 1
    kkmi = max(int(n*qmi),minbin)


    y   = zeros(nbmax)
    ibi = zeros(nbmax)
    p   = zeros((nf,nb+kma)) # Pad out for wrapped phase

    bpow = 0.
    x -= np.mean(x)

    #=====================================#
    # Triple-nested loop over   index     #
    # 1. Period                 jf        #
    # 2. Phase of transit       i (weave) # 
    # 3. Transit Duration       j (weave) #
    #=====================================#

    farr = np.linspace(fmin,fmin+nf*df,nf) 
    bins = np.linspace(0,1,nb+1)

    for jf in range(nf):
        f0 = farr[jf]

#       Zero-out working arrays
        y   = zeros(nbmax).astype(float)
        ibi = zeros(nbmax).astype(int)

#       Phase fold the data according to the trial period.
        ph = t*f0
        ph = np.mod(ph,1.)

#       Put the data in bins.
        y = ( np.histogram(ph,bins=bins,weights=x) )[0]
        ibi = ( np.histogram(ph,bins=bins) )[0]

#       EEBLS extend y and ibi so they include kma more points
        y   = np.append(y,y[0:kma])
        ibi = np.append(ibi,ibi[0:kma])

#       Loop over phase and tdur.
        p[jf,::] = blsw_loop2ph(y,ibi,kma,kmi,kkmi)


    ph = bins[:-1]
    ph = np.append(ph,ph[0:kma]+1)
    return p,farr,ph


def blsw_loop2ph(y,ibi,kma,kmi,kkmi):
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
    
    # power is now an array.
    power = np.zeros(nb) 

    # TODO: opening the file inside the loop is inefficient
    fid = open('ccode/blsw_loop2ph.c') 
    code = fid.read()
    fid.close()


    weave.inline(code,
                 ['y','ibi','kma','kmi','kkmi','rn','nb','power'],
                 type_converters=converters.blitz)
    return power



def blsmod(t,x,nf,fmin,df,nb,qmi,qma,n):
    """
    Python BLS - a modular version of BLS

    Fits five parameters:
       - P_0 - Fundemental period
       - q   - fractional time of period in transit
       - L   - low value
       - H   - high value
       - t_0 - epoch of transit       

    The output is called signal residue
    """
    minbin = 5
    nbmax = 2000

    u = zeros(n)
    v = zeros(n)

    y   = zeros(nbmax)
    ibi = zeros(nbmax)
    p   = zeros(nf)

    # Number of bins specified by the user cannot exceed hard-coded amount
    if nb > nbmax:
        print ' NB > NBMAX !!'
        return None

    # tot is the time baseline
    tot = t[-1] - t[0]
    # The minimum frequency must be greater than 1/T
    # We require there be one dip.
    # To nail down the period, wouldn't we want 2 dips.

    if fmin < 1./tot:
        print ' fmin < 1/T !!'
        return None

#------------------------------------------------------------------------
#     turn n into a float
    rn = float(n)

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

    bpow = 0.

#=================================
#     Set temporal time series
#=================================

    s = 0.

#  Subtract start time and mean
    
    u = t-t[0]
    v = x-np.mean(x)

#==============================
#     Triple-nested loop over
#     1. Period
#     2. Phase of transit
#     3. Transit Duration
#==============================

    for jf in range(nf):
        # jf - index for stepping through trial frequencies
        f0 = fmin+df*float(jf) 
        p0 = 1./f0 # p0 - period corresponding to the trial frequency

#======================================================
#     Compute folded time series with  *p0*  period
#======================================================
#     Zero-out working arrays
        y = zeros(nbmax).astype(float)
        ibi = zeros(nbmax).astype(int)

#     Phase fold the data according to the trial period
        for i in range(n):
              
# ph is the "phase" of the ith measurement (note their is a 2pi missing)
            ph = u[i]*f0
#     modulo(phase,period)
            ph = ph-int(ph)
#     j what bin is the ith data point in?
            j  = int(nb*ph)
#     ibi counts the number of points in the bin
            ibi[j] = ibi[j] + 1
#     y sums the points 
            y[j] = y[j] + v[i]
            

        power,jn1,jn2,rn3,s3 = blsmod_loop2(y,ibi,kma,kmi,kkmi)

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


def blsmod_loop2(y,ibi,kma,kmi,kkmi):
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
    power = 0

    for i in range(nb):
        s     = 0.
        k     = 0
        kk    = 0
        nb2   = i+kma
        if nb2 >= nb:
            nb2=nb-1

# Loop over the duration of transit.
        jarr = linspace(i,nb2,nb2-i+1).astype(int)
        for j in jarr:
            k = k+1
            kk = kk+ibi[j]
            s = s+y[j]

# Which (duration,phase) combination that produces peak power?
            if (k > kmi) & (kk > kkmi):
                rn1 = float(kk)
                pow = s*s/(rn1*(rn-rn1))

                if pow > power:
                    power = pow
                    jn1   = i
                    jn2   = j
                    rn3   = rn1
                    s3    = s

    return power,jn1,jn2,rn3,s3


########################
#  PLAYING WITH WEAVE  #
########################

def sum(x):
    s=zeros(1)
    code = r"""
for (int i=0;i<Nx[0];i++)
{
   s(0)= s(0)+x(i);
}
"""
    weave.inline(code,['x','s'],type_converters=converters.blitz)
    return s

def loop1(t,x):
   
    n = len(t)
    u = np.zeros(n)
    s = np.zeros(1);

    code = """
double s=0;
for (int i=0;i<Nt[0];i++)
{
   u(i) = t(i) - t(0);
   s(0))=s+x(i);
}
"""

    weave.inline(code,['u','t','s','x'],type_converters=converters.blitz)
    return s,u




def c_multi_return(x):
    code = """
printf("%i",Nx[0]);
"""
    return weave.inline(code,['x'],type_converters=converters.blitz)


def pos(x):
    """
    Simple test of conditional expressions in C

    return 1 if positive else, 0.
    """


    code = """
int pos;
if (x < 0)
 {
   pos = 0;
 }
else
 {
   pos = 1;
 } 
   return_val = pos;
"""
    return weave.inline(code,['x'],type_converters=converters.blitz)



def test_and(x,y):
    """
    Simple test of boolean operators in C

    return 1 if x and y are positive
    """
    code = """
int pos;
if ( (x > 0) && (y > 0) )
 {
   pos = 1;
 }
else
 {
   pos = 0;
 } 
   return_val = pos;
"""
    return weave.inline(code,['x','y'],type_converters=converters.blitz)




def f2i(x):
    """
    turn a float to an integer
    """
    code = """
return_val = (int) x;
"""
    return weave.inline(code,['x'],type_converters=converters.blitz)

def test_sqrt(x):
    """
    turn a float to an integer
    """
    code = """
return_val = sqrt(x);
"""
    return weave.inline(code,['x'],type_converters=converters.blitz)


def i2f(x):
    """
    turn an int to an float
    """
    code = """
return_val = (double) x;
"""
    return weave.inline(code,['x'],type_converters=converters.blitz)

