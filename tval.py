"""
Transit Validation

After the brute force period search yeilds candidate periods, functions in this module will check for transit-like signature.

"""
import numpy as np
from numpy import ma,tanh,sin,cos,pi
from numpy.polynomial import Legendre
from scipy import optimize
import scipy.ndimage as nd
import detrend
import sys

import atpy

import qalg
from keptoy import lc,a2tdur,P2a,inject
import tfind

c = 100   # Parameter that controls how sharp the transit edge is.


def pd2arr(p):
    return np.array([ p['P'],p['epoch'],p['df'],p['tdur'] ])


def model(p,t):
    P     = p[0]
    epoch = p[1]
    df    = p[2]
    tdur  = p[3]

    fmod = inject(t,np.zeros(t.size),P=p[0],epoch=p[1],df=p[2], tdur=p[3] )
    return fmod

def err(p,time,fdt):
    fmod = model(p,time)
    return (((fmod - fdt)/1e-4)**2).sum()

def objP05(p,time,fdt):
    fmod = P05(p,time)
    return (((fmod - fdt)/1e-4)**2).sum()

def model1T(p,t,P):
    pP05 = [P , p[0] , p[1] , p[2] ]
    pLeg = p[3:]
    domain = [t.min(),t.max()]

    signal = P05(pP05,t)
    trend  = Legendre(pLeg,domain=domain)  ( t )
    return signal + trend

def obj1T(p,t,f,P):
    """
    Objective for 1 Transit.
    
    Objective function for fitting a single transit.

    p - List of parameters.
    epoch = p[0]
    df    = p[1]
    tdur  = p[2]
    Legendre coeff: p[3:]
    P - period

    Period is held fixed since for a single transit, it doesnot make
    sense to specify period and epoch
    """
    model = model1T(p,t,P)
    resid  = (model - f)/1e-4
    return (resid**2).sum()

def myfunct(p, fjac=None, x=None, y=None, err=None):
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    status = 0
    return ([0, (y - P05(p,x) )/err])

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

def LDT(t,f,p,wd=2.):
    """
    Local detrending.  

    At each putative transit, fit a model transit and continuum lightcurve.

    Parameters
    ----------

    t  : Times (complete data string)
    f  : Flux (complete data string)
    p  : Parameters {'P': , 'epoch': , 'tdur': }

    """
    P     = p['P']
    epoch = p['epoch']
    tdur  = p['tdur']

    twd = round(tdur/lc)

    Pcad     = int(round(P/lc))
    epochcad = int(round(epoch/lc))
    wdcad    = int(round(wd/lc))
    tIntrp,fIntrp = detrend.nanIntrp(t,f,nContig=100/lc)

    bK,boxK,tK,aK,dK = tfind.GenK( twd ) 
    dM,bM,aM,DfDt,f0 = tfind.MF(fIntrp,twd)

    fn = ma.masked_invalid(f)
    bgood = (~fn.mask).astype(float) # 1. if the data is good (non a nan)

    bfl = nd.convolve(bgood,bK) # Filling factor.  
    afl = nd.convolve(bgood,aK)
    tfl = nd.convolve(bgood,tK)

    fl = (bfl > 0.25) & (afl > 0.25) & (tfl > 0.25) 

    f0W = tfind.XWrap(f0,Pcad,fill_value=np.nan)
    dMW = tfind.XWrap(dM,Pcad,fill_value=np.nan)
    flW = tfind.XWrap(fl,Pcad,fill_value=False)

    ### Determine the indecies of the points to fit. ###
    ms   = np.arange( f0W.shape[0] ) * Pcad + epochcad

    # Exclude regions where the convolution returned a nan.
    ms   = [m for m in ms if flW.flatten()[m] ]
    sLDT = [slice(m - wdcad/2 , m+wdcad/2) for m in ms]

    # Detrended time and flux arrays.  The only unmasked points
    # correspond to viable transits
    tdt = ma.masked_array(t,mask=True)
    fdt = ma.masked_array(f,copy=True,mask=True)
    trend = ma.masked_array(f,copy=True,mask=True)

    p1L = []
    # Fit each segment individually and store best fit parameters.
    for s,m in zip(sLDT,ms):
        y = ma.masked_invalid(f[s]) 
        x = ma.masked_array(t[s],mask=y.mask)

        x = x.compressed()
        y = y.compressed()


        df0 = dMW.flatten()[m] # Guess for transit depth
        
        p0 = [epoch, df0, tdur, f0[m], DfDt[m] , 0 ,0]

        p1, fopt ,iter ,funcalls, warnflag = \
            optimize.fmin(obj1T,p0,args=(x,y,P),maxiter=10000,maxfun=10000,
                          full_output=True,disp=False)

        domain = [x.min(),x.max()]
        trend[s] = Legendre(p1[3:],domain=domain)  ( t[s] )
        
        trend.mask[s] = False # Unmask the transit segments
        tdt.mask[s]   = False
        fdt.mask[s]   = False

        fdt[s] = f[s] - trend[s]    
        p1L.append(p1)


    fdt = ma.masked_invalid(fdt)
    tdt.mask = fdt.mask

    return tdt,fdt,p1L


def fitcand(t,f,p0,disp=True):
    """

    Fit Candidate Transits

    Starting from the promising (P,epoch,tdur) combinations returned by the
    brute force search, perform a non-linear fit for the transit.

    Parameters
    ----------

    t      : Time series  
    f      : Flux
    p0     : Dictionary {'P':Period,'epoch':Trial epoch,'tdur':Transit Duration}

    """
    twdcad = 2./lc
    P = p0['P']
    epoch = p0['epoch']
    tdur = p0['tdur']


    try:
        tdt,fdt,p1L = LDT(t,f,p0)
        nT = len(p1L)
        dtpass = True
    except:
        print sys.exc_info()
        nT = 0 
        dtpass = False
        
        
#    import pdb;pdb.set_trace()
    p0 = np.array([P,epoch,0.e-4,tdur])
    fitpass = False
    if (nT >= 3) and dtpass :
        try:
            tfit = tdt.compressed() # Time series to fit 
            ffit = fdt.compressed() # Flux series to fit.

            p1 = optimize.fmin(objP05,p0,args=(tfit,ffit),disp=False)
            tfold = tfind.getT(tdt,p1[0],p1[1],p1[3])
            fdt2 = ma.masked_array(fdt,mask=tfold.mask)
            print fdt2.count()
            print p1
            if fdt2.count() > 20:
                s2n = - ma.mean(fdt2)/ma.std(fdt2)*np.sqrt( fdt2.count() )
                fitpass = True
            else: 
                fitpass = False
                s2n = 0
            if disp:
                print "%7.02f %7.02f %7.02f" % (p1[0] , p1[1] , s2n )
        except:
            print sys.exc_info()

    if fitpass:
        return dict( P=p1[0],epoch=p1[1],df=p1[2],tdur=p1[3],s2n=s2n )
    else:
        return dict( P=p0[0],epoch=p0[1],df=p0[2],tdur=p0[3],s2n=0 )



def fitcandW(t,f,dL,par=False):
    """
    """

    n = len(dL)
    if par:
        from IPython.parallel import Client
        rc = Client()
        lview = rc.load_balanced_view()
        resL = lview.map(fitcand , n*[t] , n*[f] , dL, n*[False] ,block=True)
    else:
        resL = map(fitcand , n*[t] , n*[f] , dL)
 
    return resL



def parGuess(s2n,PG,epoch,nCheck=50):
    """
    Parameter guess

    Given the results of the matched filter approach, return the guess
    values for the non-linear fitter.

    Parameters
    ----------

    s2n   : Array of s2n
    P     : Period grid
    epoch : Array of epochs
    
    Optional Parameters
    -------------------

    nCheck : How many s2n points to look at?

    Notes
    -----

    Right now the transit duration is hardwired at 0.3 days.  This it
    should take the output value of the matched filter.

    """

    idCand = np.argsort(-s2n)
    dL = []
    for i in range(nCheck):
        idx = idCand[i]
        d = dict(P=PG[idx],epoch=epoch[idx],tdur=0.3)
        dL.append(d)

    return dL


import matplotlib.pylab as plt

def iPoP(tset,tabval):
    """
    """
    
    nsim = len(tset.PAR.P)
    print "sim, iP    ,   oP   ,  eP , iepoch,oepoch,eepoch, s2n"

    tl,fl = qalg.genEmpLC(qalg.tab2dl(tset.PAR),tset.LC.t,tset.LC.f)
    for isim in range(nsim):
        s2n = ma.masked_invalid(tabval[isim].s2n)
        iMax = s2n.argmax()

        s2n  = tabval[isim].s2n[iMax]

        iP =  tset.PAR.P[isim]
        oP =  tabval[isim].P[iMax]
        
        iepoch = tset.PAR.epoch[isim]
        oepoch = tabval[isim].epoch[iMax]

#        fig,axL = plt.subplots(2,1)
#
#        ip = dict(P=iP,epoch=iepoch,tdur=2/lc)
#        op = dict(P=oP,epoch=oepoch,tdur=2/lc)
#
#        tdt,ifdt,p1L = LDT(tl[isim],fl[isim],ip)
#        fig.add_subplot
#        axL[0].plot(ifdt)
#
#        tdt,ofdt,p1L = LDT(tl[isim],fl[isim],op)
#        axL[1].plot(ofdt)
#
#        p0 = [oP,oepoch,0,0.3]
#        p1 = optimize.fmin(err,p0,args=(tdt,ofdt),disp=False)
#        axL[1].plot(model(p1,tdt))
#
#        fig.savefig('sketch/iPoP%03i.png' % isim)

        if s2n > 5:
            print "%03i %.2f  %.2f  %+.2f  %.2f  %.2f  %+.2f  %.2f" % \
                (isim,iP,oP,100*(iP-oP)/iP, iepoch,oepoch,iepoch-oepoch ,s2n)
        else:
            print "%03i ------  ------  -----  -----  -----  -----  %.2f" % \
                (isim,s2n)



def tabval(file,par=False):
    """
    
    """

    file = 'test.fits'
    tset = atpy.TableSet(file)
    nsim = len(tset.PAR.P)
    tres = tset.RES
    
    # Check the 50 highest s/n peaks in the MF spectrum

    tabval = atpy.TableSet()
    tl,fl = qalg.genEmpLC(qalg.tab2dl(tset.PAR),tset.LC.t,tset.LC.f)
    for isim in range(nsim):
        s2n   = tres.s2n[isim]
        P     = tres.PG[isim]
        epoch = tres.epoch[isim]
        dL = parGuess(s2n,P,epoch,nCheck=50)
        resL = fitcandW(tl[isim],fl[isim],dL,par=par)

        print 21*"-" + " %d" % (isim)
        print "  iP   oP    s2n    "
        for d,r in zip(dL,resL):
            print "%7.02f %7.02f %7.02f" % (d['P'],r['P'],r['s2n'])

        tab = qalg.dl2tab(resL)

    fileL = file.split('.')
    tabval.write(fileL[0]+'_val'+'.fits',overwrite=True)


def nlfit(t,f,tres,isim):
    """

    """

    print isim
    resL = []
    for d in dL:
        p1,s2n = tval.fitcand(t,f,d)
        resL.append( dict(P=p1[0],epoch=p1[1],tdur=p1[3],s2n=s2n) )


    tab.table_name = 'sim%02i' % isim
    return tab







