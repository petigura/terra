"""
Transit Validation

After the brute force period search yeilds candidate periods, functions in this module will check for transit-like signature.

"""
import numpy as np
from numpy import ma

from scipy import optimize
import scipy.ndimage as nd
import detrend
import sys
import copy
import glob

import atpy
import qalg
import keptoy
from keptoy import lc
import tfind

import matplotlib.pylab as plt

def trsh(P,tbase):
    ftdurmi = 0.5
    tdur = keptoy.a2tdur( keptoy.P2a(P) ) 
    tdurmi = ftdurmi*tdur 
    dP     = tdurmi * P / tbase
    depoch = tdurmi

    return dict(dP=dP,depoch=depoch)

def objMT(p,time,fdt,p0,dp0):
    """
    Multitransit Objective Function

    With a prior on P and epoch

    """
    fmod = keptoy.P05(p,time)
    resid = (fmod - fdt)/1e-4
    obj = (resid**2).sum() + (((p0[0:2] - p[0:2])/dp0[0:2])**2 ).sum()
    return obj

def obj1T(p,t,f,P,p0,dp0):
    """
    Single Transit Objective Function
    """
    model = keptoy.P051T(p,t,P)
    resid  = (model - f)/1e-4
    obj = (resid**2).sum() + (((p0[0:2] - p[0:2])/dp0[0:2])**2 ).sum()
    return obj

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
    ffit = ma.masked_array(f,copy=True,mask=True)

    p1L = []
    # Fit each segment individually and store best fit parameters.
    for s,m in zip(sLDT,ms):
        y = ma.masked_invalid(f[s]) 
        x = ma.masked_array(t[s],mask=y.mask)

        x = x.compressed()
        y = y.compressed()

        df0 = dMW.flatten()[m] # Guess for transit depth
        
        p0 = [epoch, df0, tdur, f0[m], DfDt[m] , 0 ,0]

        tbase = t.ptp()
        dp0 =  trsh(P,tbase)
        dp0 = [dp0['dP'],dp0['depoch']]

        p1, fopt ,iter ,funcalls, warnflag = \
            optimize.fmin(obj1T,p0,args=(x,y,P,p0,dp0),maxiter=10000,maxfun=10000,
                          full_output=True,disp=False)

        trend[s] = keptoy.trend(p1[3:], t[s])
        ffit[s] =  keptoy.P051T(p1, t[s],P)
        
        trend.mask[s]  = False # Unmask the transit segments
        tdt.mask[s]    = False
        fdt.mask[s]    = False
        ffit.mask[s]   = False

        fdt[s] = f[s] - trend[s]    
        p1L.append(p1)

    fdt = ma.masked_invalid(fdt)
    tdt.mask = fdt.mask
    
    ret = dict(tdt=tdt,fdt=fdt,trend=trend,ffit=ffit,p1L=p1L)
    return ret

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
        dLDT = LDT(t,f,p0)
        tdt,fdt,p1L = dLDT['tdt'],dLDT['fdt'],dLDT['p1L']
        nT = len(p1L)
        dtpass = True
    except:
        print sys.exc_info()
        nT = 0 
        dtpass = False
                
    p0 = np.array([P,epoch,0.e-4,tdur])
    fitpass = False
    if (nT >= 3) and dtpass :
        try:
            tfit = tdt.compressed() # Time series to fit 
            ffit = fdt.compressed() # Flux series to fit.

            tbase = t.ptp()
            dp0 =  trsh(P,tbase)
            dp0 = [dp0['dP'],dp0['depoch']]
            p1 = optimize.fmin(objMT,p0,args=(tfit,ffit,p0,dp0) ,disp=False)
            tfold = tfind.getT(tdt,p1[0],p1[1],p1[3])
            fdt2 = ma.masked_array(fdt,mask=tfold.mask)
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

def fitcandW(t,f,dL,view=None):
    """
    """
    n = len(dL)
    if view != None:
        resL = view.map(fitcand , n*[t] , n*[f] , dL, n*[False] ,block=True)
    else:
        resL = map(fitcand , n*[t] , n*[f] , dL)
 
    return resL

def tabval(file,view=None):
    """
    
    """
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
        resL = fitcandW(tl[isim],fl[isim],dL,view=view)

        print 21*"-" + " %d" % (isim)
        print "   iP      oP      s2n    "
        for d,r in zip(dL,resL):
            print "%7.02f %7.02f %7.02f" % (d['P'],r['P'],r['s2n'])

        tab = qalg.dl2tab(resL)
        tab.table_name = 'SIM%03d' % (isim)
        tabval.append(tab)

    fileL = file.split('.')
    tabval.write(fileL[0]+'_val'+'.fits',overwrite=True)
    return tabval

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


thresh = 0.001


def iPoP(tset,tabval):
    """
    """
    nsim = len(tset.PAR.P)
    print "sim, iP    ,   oP   ,  eP , iepoch,oepoch,eepoch, s2n"
    tres = copy.deepcopy(tset.PAR)
    tres.add_empty_column('oP',np.float)
    tres.add_empty_column('oepoch',np.float)
    tres.add_empty_column('odf',np.float)
    tres.add_empty_column('os2n',np.float)

    tres.add_empty_column('KIC',np.int)

    for isim in range(nsim):
        s2n = ma.masked_invalid(tabval[isim].s2n)
        iMax = s2n.argmax()

        s2n  = tabval[isim].s2n[iMax]
        df  = tabval[isim].df[iMax]

        iP =  tset.PAR.P[isim]
        oP =  tabval[isim].P[iMax]
        
        iepoch = tset.PAR.epoch[isim]
        oepoch = tabval[isim].epoch[iMax]

        if s2n > 5:
            print "%03i %.2f  %.2f  %+.2f  %.2f  %.2f  %+.2f  %.2f" % \
                (isim,iP,oP,100*(iP-oP)/iP, iepoch,oepoch,iepoch-oepoch ,s2n)
        else:
            print "%03i ------  ------  -----  -----  -----  -----  %.2f" % \
                (isim,s2n)

        tres.oP[isim] = oP
        tres.oepoch[isim] = oepoch
        tres.KIC[isim] = tset.LC.keywords['KEPLERID']

        tres.os2n[isim] = s2n
        tres.odf[isim] = df


    return tres

def redSim(files):
    """
    Collects the information from each simulation and reduces it into
    1 file.
    """

    vfiles = []
    for i in range(len(files)):
        basename = files[i].split('.')[-2]
        vfiles.append(basename+'_val.fits')
        assert len(glob.glob(vfiles[i])) == 1, "val file must be unique"


    dL = []
    for f,v in zip(files,vfiles):
        print "Reducing %s and %s" % (f,v) 
        tset   = atpy.TableSet(f)
        tabval = atpy.TableSet(v)

        tres   = iPoP(tset,tabval)
        dL = dL + qalg.tab2dl(tres)
        
    tres = qalg.dl2tab(dL)  
    return tres



