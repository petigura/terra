import numpy as np
from numpy import pi,sqrt,where,std,ma,array
from numpy import ma,tanh,sin,cos,pi
from numpy.polynomial import Legendre
import copy
from transit import occultsmall

from config import Rsun,Msun,G,AU,lc,stefan_boltzmann

def a2tdur(a0,Rstar=1.,Mstar=1.):
    """
    Calculate the duration of transit assuming:
    
    - Solar radius
    - Solar mass
    - Radius of planet is negliable

    Given radius (AU)
    """
    a0 = array(a0)
    a = a0.copy()
    a *= AU # cm
    
    M = Mstar*Msun
    R = Rstar*Rsun

    P = sqrt(4*pi**2*a**3/G/M) # seconds    
    P /= 3600*24. # period in days    
    tdur = R * P / pi /a  # days
    return tdur

def P2a(P0,Mstar=1):
    """
    Period to Semi-major axis. Kepler's 3rd law.

    Parameters 
    ----------
    P0    : orbital period [days]
    Mstar : Stellar mass [solar masses]
    
    Returns
    -------
    a : semi-major axis [AU]
    """

    P0 = array(P0)
    P  = P0.copy()
    P  *= 3600*24 # seconds

    M = Mstar*Msun
    a = (G*M / (4*pi**2) * P**2)**(1/3.) # cm
    a /= AU # [AU]
    return a

def IncidentFlux(Rstar0,teff,a0):
    """
    Incident Flux
    
    Parameters
    ----------
    
    Rstar : Stellar radius [Rsun]
    teff  : Effective temperature [K]
    a     : Semi-major axis [AU]
    """
    # Convert into cgs
    Rstar = array(Rstar0).copy()
    Rstar *= Rsun # Radius in cm

    a = array(a0).copy()
    a *= AU # distance in cm

    Fp = Rstar**2 * stefan_boltzmann * teff**4 * a**-2
    return Fp

def ntpts(P,tdur,tbase,cad):
    # Compute number of points in transit
    
    # number of transits
    N = tbase / P
    
    # points per transit
    pptrans = tdur / cad

    # number of points in transit
    return N * pptrans


def at(f,t,P,phase,num,s2n):
    """
    Add a transit:

    """
    # Time of ingress
    tI = P * (phase + num)
    tdur = a2tdur(P2a(P))
    if (tI > t[-1]) | (tI < t[0]) :
        print " outside of acceptable domain"

    tid = where( (t > tI) & (t < tI+tdur) )[0]

    f[tid] = f[tid] - s2n * std(f)/sqrt( ntpts(P,tdur,t[-1]-t[0],30./60./24.) ) 

    return f

def inject(time0,flux0,**kw):
    """
    Inject a transit into an existing time series.

    Parameters
    ----------
    time0    : time series (required).
    flux0    : flux series (required).
    
    phase : Phase of ingress (phase * P = time of ingress) or:
    epoch : Epoch of mid transit.

    df    : Depth of transit (ppm)
    s2n   : Signal to noise (noise computed in a naive way).

    tdur : Transit duration (units of days).  If not given, compute
           assuming Keplerian orbit

    Returns
    -------
    f     : the modified time series.
    """
    t = time0.copy()
    f = flux0.copy()

    assert kw.has_key('epoch') ^ kw.has_key('phase') ,\
        "Must specify epoch xor phase"

    assert kw.has_key('s2n') ^ kw.has_key('df') ,\
        "Must specify s2n xor df"

    assert kw.has_key('P') , "Must specify Period"

    P = kw['P']
    
    if kw.has_key('tdur'):
        tdur = kw['tdur']
    else:
        tdur = a2tdur( P2a(P) )

    tbase = ma.ptp(t)
    tfold = np.mod(t,P)
 
    if kw.has_key('s2n'):
        noise = ma.std(f)
        df = s2n * noise /  np.sqrt( ntpts(P,tdur,tbase,lc) )
    else:
        df = 1e-6*kw['df']

    if kw.has_key('epoch'):
        epoch = kw['epoch']
    else:
        epoch = kw['phase']*P

    epoch = np.mod(epoch,P)
    tm = abs( tfold - epoch ) # Time before (or after) midtransit.
    ie = 0.5*(tdur - lc)
    oe = 0.5*(tdur + lc)

    frac = ( tm - ie ) / lc 

    idlo = np.where(tm < ie )
    idfr = np.where( (tm > ie ) & (tm < oe) )
    f[idlo] -= df
    f[idfr] -= df*(1-frac[idfr])

    return f

def ntrans(tbase,P,phase):
    """
    The expected number of transits is a minimum floor(tbase/P), 
    if the phase works out there can be 1 more.
    """
    
    tbase = float(tbase)
    P = float(P)
    return np.floor(tbase/P) + (phase <  np.mod(tbase,P) / P ).astype(np.int)


def box(p,t):
    P     = p[0]
    epoch = p[1]
    df    = p[2]
    tdur  = p[3]

    fmod = inject(t,np.zeros(t.size),P=p[0],epoch=p[1],df=p[2], tdur=p[3] )
    return fmod


def trap(p,t):
    """
    Trapezoid transit model

    Parameters
    ----------

    p : list of transit paramters
        p[0] - depth
        p[1] - duration of flat bottom
        p[2] - duration of the sloping wings

    t : time 
    """
    df   = p[0]
    fdur = p[1] # Flat duration
    wdur = p[2] # Wing duration
    f    = np.zeros(t.size)
    abst = np.abs(t)
    bw   = (abst > fdur/2) & ( abst < fdur/2 + wdur)
    bf   = (abst < fdur/2) 
    m    = df/wdur
    f[bw]= m*(abst[bw] - fdur/2) - df
    f[bf]= -df
    return f


def P05(p,t):
    """
    Analytic model of Protopapas 2005.
    """
    P     = p[0]
    epoch = p[1]
    df    = p[2]
    tdur  = p[3]

    tp = tprime(P,epoch,tdur,t)

    return df * ( 0.5*( 2. - tanh(P05c*(tp + 0.5)) + tanh(P05c*(tp - 0.5)) ) - 1.)

def tprime(P,epoch,tdur,t):
    """
    t' in Protopapas
    """
    return  ( P*sin( pi*(t - epoch) / P ) ) / (pi*tdur)


def trend(p,t):
    """
    Fit the local transit shape with the following function.

    """
    domain = [t.min(),t.max()]
    return Legendre(p,domain=domain)(t)
    
def P051T(p,t):
    """
    Protopapas single transit

    Period is held fixed since for a single transit, it doesnot make
    sense to specify period and epoch

    """
    P = 100
    pP05 = [P , p[0] , p[1] , p[2] ]
    return  P05(pP05,t)

def genEmpLC(d0,t,f):
    """
    Generate Synthetic Lightcurves

    Parameters
    ----------
    darr : List of dictionaries specifying the LC parameters.
    t    : Input time series
    f    : Input flux series


    Returns
    -------
    f :  flux arrays
    """
    d = copy.deepcopy(d0)
    f = inject(t,f,**d)
    f = f.astype(np.float32)
    return f

def genSynLC(darr):
    """
    Generate Synthetic Lightcurves

    Parameters
    ----------
    darr : List of dictionaries specifying the LC parameters.

    Returns
    -------
    tl   : List of time arrays
    fl   : List of flux arrays
    
    """
    fl,tl = [],[]
    for d in darr:
        f,t = lightcurve(**d) 
    
        f = f.astype(float32)
        t = t.astype(float32)
        fl.append(f)
        tl.append(t)
        
    return tl,fl

def occultsmall_py(p,c1,c2,c3,c4,z):
    """
    Mandel Agol light curve (small planet approx).

    This routine approximates the lightcurve for a small planqet. (See
    section 5 of Mandel & Agol (2002) for details).  Please cite
    Mandel & Agol (2002) if making use of this routine.

    Parameters
    ----------
    p       : ratio of planet radius to stellar radius       
    c1-c4   : non-linear limb-darkening coefficients                  
    z       : impact parameters (normalized to stellar radius)

    Returns
    -------
    mu      : flux relative to unobscured source for each z 
    
    Notes
    -----
    Translation of IDL routine
    """

    nb=z.size
    mu=np.ones(nb)
    # edges overlap
    bedge =  (z > 1.-p) & (z < 1.+p) 

    i1=1.-c1-c2-c3-c4

    norm = np.pi*(1.-c1/5.-c2/3.-3.*c3/7.-c4/2.)
    sig  = np.sqrt( np.sqrt( 1. - (1.-p)**2 ) )
    x    = 1.-(z[bedge]-p)**2

    tmp  = (1.-c1*(1.-4./5.*x**0.25) - c2*(1.-2./3.*x**0.5) - \
            c3*(1.-4./7.*x**0.75)-c4*(1.-4./8.*x))

    mu[bedge] = 1. - tmp*(p**2*np.arccos((z[bedge]-1.)/p) - 
                          (z[bedge]-1.)*np.sqrt(p**2-(z[bedge]-1.)**2))/norm

    # inside disk
    bin =  (z < 1.-p) & (z != 0.) 
    mu[bin] = 1.-np.pi*p**2*iofr(c1,c2,c3,c4,z[bin],p)/norm

    # div by 0 special case
    b0 = z == 0.
    mu[b0] =1. - np.pi*p**2/norm
    return mu

def iofr(c1,c2,c3,c4,r,p):
    """
    Helper function to occult small
    """
    sig1=np.sqrt(np.sqrt(1.-(r-p)**2))
    sig2=np.sqrt(np.sqrt(1.-(r+p)**2))
    
    res = 1.-c1*(1.+(sig2**5-sig1**5)/5./p/r) - \
        c2*(1.+(sig2**6-sig1**6)/6./p/r) - \
        c3*(1.+(sig2**7-sig1**7)/7./p/r) - \
        c4*(p**2+r**2)
    return res

def occultsmallt_py(t,p,c1,c2,c3,c4,tau,b):
    z =  np.sqrt( (t/tau)**2 + b**2 )
    mu =  occultsmall_py(p,c1,c2,c3,c4,z)
    return mu

def occultsmallt_fort(t,p,c1,c2,c3,c4,tau,b):
    z  =  np.sqrt( (t/tau)**2 + b**2 )
    mu =  np.empty(z.shape)
    occultsmall.occultsmall(p,c1,c2,c3,c4,z,mu)
    return mu

def genMALC(d):
    # Covert P, R* to tau.

    a = (G*Mstar*P / (4*pi**2))**(1/3.)
    n = 2*pi / P

def MA(pL,climb,t,usamp=11,occultsmallt=occultsmallt_fort):
    """
    Mandel Agol Model

    Small planet approximation given.  Transit parameters and
    limb-darkening parameters.  I compute the average intensity for a
    29.4 min bin which acconts for the convolution effects seen in
    really short transits.

    Parameters
    ----------
    pL   : transit parameters
           pL[0] - p   - Rp/R*
           pL[1] - tau - R*/an.  The time it takes the planet to
                         travel R*.  Half of the total transit
                         duration assuming b = 0
           pL[2] - b   - impact parameter.  Ranges from 0 to 1.
    climb : 4 component limb darkening model
    t     : time t0 = 0 
    occultsmallt : Which version of occultsmallt to call. Python or fortran.
    
    Returns
    -------
    fmod : Mandel Agol model convolved with the Kepler observing cadence.

    """
    p   = pL[0]
    tau = pL[1]
    b   = pL[2]

    c1,c2,c3,c4 = climb
    if usamp is 0:
        fmod = occultsmallt(t,p,c1,c2,c3,c4,tau,b)
    else:
        dt     = np.linspace( -lc/2 , lc/2 , usamp)
        tblock = t[:,np.newaxis] + dt

        # super resolution model of transit
        fblock = occultsmallt(tblock.flatten(),p,c1,c2,c3,c4,tau,b)  
        fblock = fblock.reshape(tblock.shape)
        fmod   = fblock.mean(axis=1)
    return fmod - 1

def MAfast(pL,climb,t,**kwargs):
    """
    Mandel Agol Fast

    Simple and dumb way to increase the speed of MA light curve
    fitting.  At run time I compute the full MA model for 200 points
    inside the transit.  Then I interpolate over them.
    """
    tmi,tma = t.min(),t.max()
    densew = 2*pL[1]
    tdense = np.linspace(-densew,densew,200)
    tg   = np.hstack([np.linspace(tmi,-densew,10) ,tdense,np.linspace(densew,tma,10)])
    yg   = MA(pL,climb,tg,usamp=11)
    yint = np.interp(t,tg,yg)

    return yint

import tval

def synMA(d,t0):
    """
    Inject MA light curve into light curve

    Parameters
    ----------

    d : dictionary with the transit parameters
        - Mstar
        - Rstar
        - df
        - b
        - P
        - phase
        - a1, ... ,a4
    t0 : time
    f : initial photometry
    """
    t = t0.copy()
    pL = [ d[k] for k in ['p','tau','b'] ]
    climb = np.array([d['a%i' %i] for i in range(1,5)])
    P = d['P']
    dt = tval.t0shft(t,P,d['phase']*P)
    t += dt
    tPF = np.mod(t+P/2,P)-P/2
    ft = MA(pL,climb,tPF)
    return ft

def tdurMA(d):
    """
    Transit duration for Mandel Agol LC

    Given the standard dictionary with the MA parameters, return the
    length of the transit (first contact to last contact).

    Parameters
    ----------
    d - dictionary with MA LC parameters.
    """
    
    t = np.linspace(-2,2,200)
    ft = synMA(d,t)
    tdur = t[ft < 0.].ptp()
    return tdur
