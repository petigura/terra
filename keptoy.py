import numpy as np
from numpy import pi,sqrt,where,std,ma,array
from numpy import ma,tanh,sin,cos,pi
from numpy.polynomial import Legendre
import copy

### Constants ###
G = 6.67e-8 # cgs
R_sun = 6.9e10 # cm
M_sun = 2.0e33 # g
lc = 0.0204343960431288
c = 100   # Parameter that controls how sharp P05 edge is.

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

def inject(t0,f0,**kw):
    """
    Inject a transit into an existing time series.

    Parameters
    ----------
    t0    : time series (required).
    f0    : flux series (required).
    
    phase : Phase of ingress (phase * P = time of ingress) or:
    epoch : Epoch of mid transit.

    df    : Depth of transit
    s2n   : Signal to noise (noise computed in a naive way).

    tdur : Transit duration (units of days).  If not given, compute
           assuming Keplerian orbit

    Returns
    -------
    f     : the modified time series.
    """
    t = t0.copy()
    f = f0.copy()

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
        df = kw['df']

    if kw.has_key('epoch'):
        epoch = kw['epoch']
    else:
        epoch = kw['phase']*P

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


def P05(p,t):
    """
    Analytic model of Protopapas 2005.
    """
    P     = p[0]
    epoch = p[1]
    df    = p[2]
    tdur  = p[3]

    tp = tprime(P,epoch,tdur,t)

    return df * ( 0.5*( 2. - tanh(c*(tp + 0.5)) + tanh(c*(tp - 0.5)) ) - 1.)

def tprime(P,epoch,tdur,t):
    """
    t' in Protopapas
    """
    return  ( P*sin( pi*(t - epoch) / P ) ) / (pi*tdur)

def dP05(p,t):
    """
    Analytic evalutation of the Jacobian.  

    Note 
    ----

    This function has not been properly tested.  I wrote it to see if
    it made the LM fitter any more robust or speedy.  I could not LM
    to work even numerically.
    """

    P     = p[0]
    epoch = p[1]
    df    = p[2]
    tdur  = p[3]

    # Argument to the sin function in tprime.
    sa = pi*(t - epoch) / P 
    tp = tprime(P,epoch,tdur,t)

    # Compute partial derivatives with respecto to tprime:
    dtp_dP     = sin( sa ) / pi / epoch + P * cos( sa ) / pi / epoch
    dtp_depoch = - tp / epoch
    dtp_dtdur  = - cos( sa ) / epoch

    # partial(P05)/partial(tprime)
    dP05_dtp = 0.5 * df * ( (tanh( c*(tp + 0.5) ) )**2 - \
                                 (tanh( c*(tp - 0.5) ) )**2)

    # Compute the partial derivatives
    dP05_dP     = dP05_dtp * dtp_dP

    # dP05_dP     = np.zeros(len(t))
    dP05_depoch = dP05_dtp * dtp_depoch
    dP05_ddf    = 0.5 * ( - tanh(c*(tp + 0.5)) + tanh(c*(tp - 0.5)) )
    dP05_dtdur  = dP05_dtp * dtp_dtdur
    
    # The Jacobian
    J = np.array([dP05_dP,  dP05_depoch , dP05_ddf, dP05_dtdur])

    return J

def trend(p,t):
    """
    Fit the local transit shape with the following function.

    """
    domain = [t.min(),t.max()]
    return Legendre(p,domain=domain)(t)
    
def P051T(p,t,P):
    """
    Protopapas single transit

    Single transit + trend

    Parameters
    ----------

    p : List of parameters.
        p[0]   epoch
        p[1]   df
        p[2]   tdur 
        p[3:]  passed to trend

    P : period

    Period is held fixed since for a single transit, it doesnot make
    sense to specify period and epoch

    """
    
    pP05 = [P , p[0] , p[1] , p[2] ]
    sig = P05(pP05,t)

    ptrend = p[3:]
    tr     = trend(ptrend,t)

    return sig + tr

def genEmpLC(d0,tdt,fdt):
    """
    Generate Synthetic Lightcurves

    Parameters
    ----------
    darr : List of dictionaries specifying the LC parameters.
    tdt  :
    fdt  :


    Returns
    -------
    tl   : List of time arrays
    fl   : List of flux arrays
    
    """

    d = copy.deepcopy(d0)

    d.pop('seed')
    d.pop('tbase')
    f = inject(tdt,fdt,**d)
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
