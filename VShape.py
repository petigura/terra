"""
Adapted the transit model in tval, if I use the trapezoidal model
alot, merge it back in to the main code-body.
"""
from matplotlib.pylab import *
import copy 
import keptoy
from scipy import optimize

class TransitModel:
    """
    TransitModel
    
    Simple class for fitting transit    
    """
    def __init__(self,t,f,ferr,pdict,
                 fixdict=dict(df=False,fdur=False,wdur=False,dt=True)):
        """
        Constructor for transit model.

        Parameters
        ----------
        t     : time array
        f     : flux array
        ferr  : errors
        climb : size 4 array with non-linear limb-darkening components
        pL    : Parameter list dictionary [ Rp/Rstar, tau, b, dt]. 
        fix   : Same keys as above, determines which parameters to fix
        """
        # Protect input arrays 
        self.t       = t.copy()
        self.f       = f.copy()
        self.ferr    = ferr.copy()
        self.pdict   = pdict
        self.fixdict = fixdict

        # Strip nans from t, y, and err
        b = np.vstack( map(np.isnan, [t,f,ferr]) ) 
        b = b.astype(int).sum(axis=0) == 0 
        b = b & (ferr > 0.)

        if b.sum() < b.size:
            print "removing %i measurements " % (b.size - b.sum())
            self.t    = t[b]
            self.f    = f[b]
            self.ferr = ferr[b]

    def register(self):
        """
        Register light curve
        
        Transit center will usually be within 1 long cadence
        measurement. We search for the best t0 over a finely sampled
        grid that spans twice the transit duration. At each trial t0,
        we find the best fitting MA model. We adopt the displacement
        with the minimum overall Chi2.

        """
        # For the starting value to the optimization, use the current
        # parameter vector.
        pdict0 = copy.copy(self.pdict)
        tau0   = pdict0['fdur']

        dt_per_lc = 3   # Number trial t0 per grid point
        dtarr = np.linspace(-2*tau0,2*tau0,4*tau0/keptoy.lc*dt_per_lc)

        def f(dt):
            self.pdict['dt'] = dt
            pdict,X2 = self.fit()
            return X2

        X2L = np.array( map(f,dtarr) )
        if (~np.isfinite(X2L)).sum() != 0:
            print "problem with registration: setting dt to 0.0"
            self.pdict['dt'] = 0.
        else:
            self.pdict['dt'] = dtarr[np.argmin(X2L)]
            print "registration: setting dt to %(dt).2f" % self.pdict
            
    def fit(self):
        """
        Fit MA model to LC

        Run the Nelder-Meade minimization routine to find the
        best-fitting MA light curve. If we are holding impact
        parameter fixed.
        """
        pL   = self.pdict2pL(self.pdict)
        res  = optimize.fmin(self.chi2,pL,disp=False,full_output=True)
        pdict = self.pL2pdict(res[0])
        X2    = res[1]
        return pdict,X2

    
    def chi2(self,pL):
        pdict   = self.pL2pdict(pL)
        f_model = self.trap(pdict,self.t)
        resid   = (self.f - f_model)/self.ferr
        X2 = (resid**2).sum()
        if abs(pdict['df'])<0.:
            X2=np.inf            
        if (pdict['fdur'] < 0) or (pdict['fdur'] > 1 ) :
            X2=np.inf
        if (pdict['wdur'] > 2) or (pdict['wdur'] < 0):
            X2=np.inf

        return X2

    def pL2pdict(self,pL):
        """
        Covert a list of floating parameters to the standard parameter
        dictionary.
        """
        pdict = {}
        i = 0 
        for k in self.pdict.keys():
            if self.fixdict[k]:
                pdict[k]=self.pdict[k]
            else:
                pdict[k]=pL[i]
                i+=1

        return pdict

    def pdict2pL(self,pdict):
        """
        Create a list of the floating parameters
        """
        pL = []
        for k in self.pdict.keys():
            if self.fixdict[k] is False:
                pL += [ pdict[k] ]

        return pL

    def trap(self,pdict,t):
        """
        Mandel Agol Model.

        Four free parameters taken from current pdict


        pL can either have
           3 parameters : p,tau,b
           2 parameters : p,tau (b is taken from self.fixb)
        """
        p = [ pdict[k] for k in 'df fdur wdur'.split() ] 
        res = keptoy.trap(p,t-pdict['dt'])
        return res


def TM_read_h5(h5):
    """
    Read in the information from h5 dataset and return TransitModel
    Instance.
    """    
    attrs = dict(h5.attrs)

    if attrs['P'] < 50:
        bPF  = h5['blc10PF0'][:]
        t    = bPF['tb']
        f    = bPF['med']
        ferr = bPF['std'] / np.sqrt( bPF['count'] )
        b1   = bPF['count']==1 # for 1 value std ==0 which is bad
        ferr[b1] = ma.median(ma.masked_invalid(bPF['std']))
    else:
        lc   = h5['lcPF0'][:]
        t    = lc['tPF']
        f    = lc['f']
        ferr = np.std( (f[1:]-f[:-1])*np.sqrt(2))
        ferr = np.ones(lc.size)*ferr
    try:
        p0 = np.sqrt(1e-6*attrs['df'])
    except:
        p0 = np.sqrt(attrs['mean'])

    pdict=dict(df=p0**2,fdur=attrs['tdur'],wdur=attrs['tdur']/2,dt=0.)
    trans = TransitModel(t,f,ferr,pdict)
    return trans
