import numpy as np
from numpy import pi,sqrt,where,std,ma,array

### Constants ###

G = 6.67e-8 # cgs
R_sun = 6.9e10 # cm
M_sun = 2.0e33 # g

lc = 30./60./24.
sc = 1./60./24.


def lightcurve(df=0.01,P=12.1,phase=0.5,cad=lc,tdur=None,
               s2n=10,tbase=90,a=None,null=False,seed=None,model=False):
    """
    generate sample lightcurve.
    """

    if tdur is None:
        a = P2a(P)
        tdur = a2tdur(a)

    npts = tbase/cad
    t = np.linspace(0,tbase,npts) 

    # Noise is constant
    noise = 1e-4

    if null:
        df = 0.
    elif model:
        noise = 0
    else:
        df = s2n * noise /  np.sqrt( ntpts(P,tdur,tbase,cad) )

    # Add in a seed for deterministic noise
    if seed != None:
        np.random.seed(seed)

    f = np.ones(npts) + noise*np.random.randn(npts)

    tidx = np.where( np.mod(t-phase*P,P) < tdur)[0]
    f[tidx] -= df

    return f,t

def a2tdur(a0):
    """
    Calculate the duration of transit assuming:
    
    - Solar radius
    - Solar mass
    - Radius of planet is negliable

    Given radius (AU)
    """
    a0 = array(a0)
    a = a0.copy()
    a *= 1.50e13 # cm
    
    P = sqrt(4*pi**2*a**3/G/M_sun) # seconds    
    P /= 3600*24 # period in days
    
    tdur = R_sun * P / pi /a  # days
    return tdur

def P2a(P0):
    """
    Given the period of planet (days) what is its semimajor axis assuming:

    - Solar mass
    """
    P0 = array(P0)
    P = P0.copy()
    P *= 3600*24 # seconds
    a = (G*M_sun / (4*pi**2) * P**2)**(1/3.) # cm
    a /= 1.50e13 # AU
    
    return a

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


def inject(t0,f0,s2n=100,P=12.1,phase=0.5,cad=lc):
    """
    Inject a transit into an existing time series.

    Returns
    f - the modified time series.
    """

    t = t0.copy()
    f = f0.copy()

    a = P2a(P)
    tdur = a2tdur(a)
    tbase = ma.ptp(t)


    noise = ma.std(f)
    df = s2n * noise /  np.sqrt( ntpts(P,tdur,tbase,cad) )
    tidx = np.where( np.mod(t-phase*P,P) < tdur)[0]
    f[tidx] -= df
    
    return f


def ntrans(tbase,P,phase):
    """
    The expected number of transits is a minimum floor(tbase/P), 
    if the phase works out there can be 1 more.
    """
    
    tbase = float(tbase)
    P = float(P)
    return np.floor(tbase/P) + (phase <  np.mod(tbase,P) / P ).astype(np.int)
