import sys

import numpy as np
from scipy import optimize
import pandas as pd
from matplotlib.pylab import *
from emcee import EnsembleSampler

import keptoy
from plotting.kplot import *
from utils import h5plus 

class TransitModel(h5plus.iohelper):
    """
    TransitModel
    
    Simple class for fitting transit    

    Fit Mandel Agol (2002) to phase folded light curve. For short
    period transits with many in-transit points, we fit median phase
    folded flux binned up into 10-min bins to reduce computation time.

    Light curves are fit using the following proceedure:
    1. Registraion : best fit transit epcoh is determined by searching
                     over a grid of transit epochs
    2. Simplex Fit : transit epoch is frozen, and we use a simplex
                     routine (fmin) to search over the remaining
                     parameters.
    3. MCMC        : If runmcmc is set, explore the posterior parameter space
                     around the best fit results of 2.

    Parameters
    ----------
    h5     : h5 file handle

    Returns
    -------
    Modified h5 file. Add/update the following datasets in h5['fit/']
    group:

    fit/t     : time used in fit
    fit/f     : flux used in fit
    fit/fit   : best fit light curve determined by simplex algorithm
    fit/chain : values used in MCMC chain post burn-in

    And the corresponding attributes
    fit.attrs['pL0']  : best fit parameters from simplex alg
    fit.attrs['X2_0'] : chi2 of fit
    fit.attrs['dt_0'] : Shift used in the regsitration.
    """

    def __init__(self,t,f,ferr,climb,pdict,
                 fixdict=dict(p=False,tau=False,b=False,dt=True)):
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
        super(TransitModel,self).__init__()

        # Protect input arrays 
        t = t.copy()
        f = f.copy()
        ferr = ferr.copy()

       # Strip nans from t, y, and err
        b = np.vstack( map(np.isnan, [t,f,ferr]) ) 
        b = b.astype(int).sum(axis=0) == 0 
        b = b & (ferr > 0.)

        if b.sum() < b.size:
            print "removing %i measurements " % (b.size - b.sum())
            t = t[b]
            f = f[b]
            ferr = ferr[b]

        self.climb   = climb
        self.pdict   = pdict
        self.fixdict = fixdict
        self.add_dset('t',t,description='time')
        self.add_dset('f',f,description='flux')
        self.add_dset('ferr',ferr,description='error on flux')
        self.add_attr('completed_mcmc',0,
                      description='Sucessful MCMC Run? 0/1=N/Y')

    def register(self):
        """
        Register light curve
        
        Transit center will usually be within 1 long cadence
        measurement. We search for the best t0 over a finely sampled
        grid that spans twice the transit duration. At each trial t0,
        we find the best fitting MA model. We adopt the displacement
        with the minimum overall Chi2.

        """
        # Some where something is overloading the copy function with
        # the numpy copy function!!!

        import copy
        pdict0 = copy.copy(self.pdict)
        tau0   = pdict0['tau']

        dt_per_lc = 3   # Number trial t0 per grid point
        dtarr = np.linspace(-2*tau0,2*tau0,4*tau0/keptoy.lc*dt_per_lc)

        def f(dt):
            self.pdict['dt'] = dt
            pdict,X2 = self.fit_lightcurve()
            return X2

        X2L = np.array( map(f,dtarr) )
        if (~np.isfinite(X2L)).sum() != 0:
            print "problem with registration: setting dt to 0.0"
            self.pdict['dt'] = 0.
        else:
            self.pdict['dt'] = dtarr[np.argmin(X2L)]
            print "registration: setting dt to %(dt).2f" % self.pdict

    def fit_lightcurve(self):
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

    def handel_error(func):
        """
        Cut down on the number of try except statements
        """
        def wrapped(self):
            try:
                func(self)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print "%s: %s: %s" %  (func.__name__,exc_type,exc_value)
        return wrapped

    @handel_error
    def MCMC(self):
        """
        Run MCMC

        Explore the parameter space around the current pL with MCMC

        Adds the following attributes to the transit model structure:

        upL0  : 3x3 array of the 15, 50, and 85-th percentiles of
                Rp/Rstar, tau, and b
        chain : nsamp x 3 array of the parameters tried in the MCMC chain.
        fits  : Light curve fits selected randomly from the MCMC chain.
        """
        
        # MCMC parameters
        nwalkers = 10; ndims = 3
        nburn   = 1000
        niter   = 2000
        print """\
running MCMC
------------
%6i walkers
%6i step burn in
%6i step run
""" % (nwalkers,nburn,niter)

        # Initialize walkers
        pL  = self.pdict2pL(self.pdict)
        fltpars = [ k for k in self.pdict.keys() if not self.fixdict[k] ]
        allpars = self.pdict.keys()
        p0  = np.vstack([pL]*nwalkers) 
        for i,name in zip(range(ndims),fltpars):
            if name=='p':
                p0[:,i] += 1e-4*np.random.randn(nwalkers)
            elif name=='tau':
                p0[:,i] += 1e-2*pL[i]*np.random.random(nwalkers)
            elif name=='b':
                p0[:,i] = 0.8*np.random.random(nwalkers) + .1

        # Burn in 
        sampler = EnsembleSampler(nwalkers,ndims,self)
        pos, prob, state = sampler.run_mcmc(p0, nburn)

        # Real run
        sampler.reset()
        foo   = sampler.run_mcmc(pos, niter, rstate0=state)

        chain  = pd.DataFrame(sampler.flatchain,columns=fltpars)
        uncert = pd.DataFrame(index=['15,50,85'.split(',')],columns=allpars)
        for k in self.pdict.keys():
            if self.fixdict[k]:
                chain[k]  = self.pdict[k]
                uncert[k] = self.pdict[k]
            else:
                uncert[k] = np.percentile( chain[k], [15,50,85] )

        nsamp = 200
        ntrial = sampler.flatchain.shape[0]
        id = np.random.random_integers(0,ntrial-1,nsamp)

        f = lambda i : self.MA( self.pL2pdict(sampler.flatchain[i]),self.t)
        fits = np.vstack( map(f,id) )         

        uncert = uncert.to_records(index=False)
        chain = chain.to_records(index=False)

        self.add_dset('uncert',uncert,description='uncertainties')
        self.add_dset('chain',chain,description='MCMC chain')
        self.add_dset('fits',fits,description='Fits from MCMC chain')
        self.completed_mcmc = 1 # Note that MCMC was sucessful

    def __call__(self,pL):
        """
        Used for emcee MCMC routine
        
        pL : list of parameters used in the current MCMC trial
        """
        loglike = -self.chi2(pL)
        return loglike

    def chi2(self,pL):
        pdict   = self.pL2pdict(pL)
        f_model = self.MA(pdict,self.t)
        resid   = (self.f - f_model)/self.ferr
        X2 = (resid**2).sum()
        if (pdict['tau'] < 0) or (pdict['tau'] > 2 ) :
            X2=np.inf
        if (pdict['b'] > 1) or (pdict['b'] < 0):
            X2=np.inf
        if abs(pdict['p'])<0.:
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
    
    def MA(self,pdict,t):
        """
        Mandel Agol Model.

        Four free parameters taken from current pdict


        pL can either have
           3 parameters : p,tau,b
           2 parameters : p,tau (b is taken from self.fixb)
        """
        pMA3 = [ pdict[k] for k in 'p tau b'.split() ] 
        res = keptoy.MA(pMA3, self.climb, t - pdict['dt'], usamp=5)
        return res

    def get_MCMC_dict(self):
        """
        Returns a dictionary with the best fit MCMC parameters.
        """
        keys = 'p tau b'.split()
        ucrt = pd.DataFrame(self.uncert,index=['15 50 85'.split()])[keys].T
        ucrt['med'] = ucrt['50']
        ucrt['sig'] = (ucrt['85']-ucrt['15'])/2

        if ucrt.ix['b','85']-ucrt.ix['b','50'] > ucrt.ix['b','15']:
            ucrt.ix['b','sig'] =None
            ucrt.ix['b','med'] =ucrt.ix['b','85']

        # Output dictionary
        d = {}
        for k in keys:
            d[k]     = ucrt.ix[k,'med']
            d['u'+k] = ucrt.ix[k,'sig']

        return d

    def to_hdf(self,h5file,group):
        """
        TODO
        ----
        Should define these as attributes in when they are created.

        """
        self.fit = self.MA(self.pdict,self.t)
        self.add_dset('fit',self.fit,description='Best fitting light curve')
        self.add_attr('p',self.pdict['p'],
                       description='Planet star radius ratio')
        self.add_attr('tau',self.pdict['tau'],
                       description='tau Mandel Agol (2005)')
        self.add_attr('b',self.pdict['b'],
                       description='impact parameter')
        self.add_attr('dt',self.pdict['dt'],
                       description='Time of transit adjusted')
        super(TransitModel,self).to_hdf(h5file,group)
        

def read_hdf(h5file,group):
    tm = h5plus.read_iohelper(h5file,group)
    tm.__class__ = TransitModel
    return tm 

def TM_unitsMCMCdict(d0):
    """
    tau e[day] -> tau[hrs]
    p [frac] -> p [percent]
    """
    d = copy.copy(d0)
    
    d['p']    *= 100.
    d['up']   *= 100.
    d['tau']  *= 24.
    d['utau'] *= 24.
    return d 

def TM_stringMCMCdict(d0):
    """
    Convert MCMC parameter output to string
    """
    d = copy.copy(d0) 
    keys = 'p,tau,b'.split(',')    
    for k in keys:
        for k1 in [k,'u'+k]:
            d[k1] = "%.2f" % d0[k1]

    if np.isnan(d0['ub']):
        d['b']  = "<%(b)s" % d
        d['ub'] = ""

    return d

def from_dv(dv,bin_period=50):
    """
    From DataValidation Object.

    Read in necessary info from DataValidation object and return new
    transit model object.

    Parameters
    ----------
    dv : DataValidation Object
    bin_period : To speed up MCMC for light curves with many transits,
                 we can operate on binned arrays. For K2, the campaign
                 are short enough that this isn't an issue.
    """    
    if dv.P < bin_period:
        print "P < %.1f, using binned (10min) light curve" % bin_period
        bPF  = dv.blc10PF0
        t    = dv.tb
        f    = dv.med
        ferr = dv.std / np.sqrt( bPF['count'] )
        b1   = bPF['count']==1 # for 1 value std ==0 which is bad
        ferr[b1] = ma.median(ma.masked_invalid(bPF['std']))
    else:
        lcPF = dv.lcPF0
        t    = lcPF['tPF']
        f    = lcPF['f']
        ferr = np.std( (f[1:]-f[:-1])*np.sqrt(2) )
        ferr = np.ones(lcPF.size)*ferr

    # Initial guess for MA parameters.
    pdict = dict(p = np.sqrt(dv.mean),
                 tau = dv.tdur/2. ,
                 b = 0.3,
                 dt = 0.0) 

    # Find global best fit value
    trans = TransitModel(t,f,ferr,dv.climb,pdict)
    return trans



def plot_transit_model(tm):
    import pdb;pdb.set_trace()
    gs = GridSpec(4,1)        
    fig = gcf()
    axTrans = fig.add_subplot(gs[1:])
    plot_trans(tm)
#    yticksppm()
    legend()

    axRes   = fig.add_subplot(gs[0])
    plot_res(tm)
#    yticksppm()
    gcf().set_tight_layout(True)
    legend()

def plot_trans(tm):
    plot(tm.t,tm.f,'.',label='PF LC')
    plot(tm.t,tm.fit,lw=2,alpha=2,label='Fit')

def plot_res(tm):
    plot(tm.t,tm.f-tm.fit,lw=1,alpha=1,label='Residuals')        
