import pandas as pd
import numpy as np
from numpy import histogram2d as h2d
import scipy.stats
import sqlite3
from pandas.io import sql
import glob
import os
from matplotlib.pylab import *

from config import G,Rsun,Rearth,Msun,AU,sec_in_day

def pngs2txt(path):
    """
    List all the .png files in a directory and create a flat ascii of
    all the basenames.
    """
    fL = glob.glob('%s/*.png' %path)
    print "reading pngs from %s" %path
    df = pd.DataFrame(fL,columns=['file'])
    df['skic'] = df.file.apply(lambda x : x.split('/')[-1].split('.')[0])
    outfile = '%s.txt' % path
    print "writing skic to %s" % outfile
    df.skic.to_csv(outfile,index=False)


def upoisson(Np):
    """
    Poisson Uncertanty

    Returns fractional uncertanty (asymetric)
    """
    x  = np.arange(100)
    cdf = scipy.stats.poisson.cdf(x,Np)

    xi = np.linspace(0,x[-1],1000)
    yi = np.interp(xi,x,cdf)
    lo = xi[np.argmin(np.abs(yi-.159))]
    hi = xi[np.argmin(np.abs(yi-.841))]
    lo = (Np-lo) / Np
    hi = (hi-Np) / Np
    return lo,hi

def getbins(panel):
    lua = lambda x : list(np.unique(np.array(x)))
    Rpb = lua(panel.Rp1) + [ lua(panel.Rp2)[-1] ]
    Pb  = lua(panel.P1)  + [ lua(panel.P2)[-1]  ]
    bins = Pb,Rpb
    return bins

comp_ness_flds = 'inj_P,inj_Rp,comp'.split(',')
def completeness(panel,df_mc):
    """
    Compute completeness

    df_mc  : dataframe with the following columns
             - inj_P
             - inj_Rp
             - comp (whether a particular simluation counts toward
               completeness)

    """    
    for k in comp_ness_flds:
        assert list(df_mc.columns).count(k) == 1,'Missing %s' % k

    bins = getbins(panel)
    
    df_comp       = df_mc[ df_mc['comp'] ]
    nPass , xe,ye = h2d(df_comp.inj_P , df_comp.inj_Rp , bins=bins)
    nTot ,  xe,ye = h2d(df_mc.inj_P   , df_mc.inj_Rp   , bins=bins)

    panel['nPass'] = nPass
    panel['nTot']  = nTot
    panel['comp'] = nPass/nTot
    panel['comp'][np.isnan(panel)] = 0 
    return panel


def compareCatalogs(cat1,cat2,suffixes=['cat1','cat2']):
    """
    Compare Catalogs
    
    Determine which planets appear in both catalogs. Candidates are
    considered equal if:

    |P_cat1 - P_cat2| < 0.01 days

    I believe the SQL equivalent is:
    
    SELECT * FROM cat1 OUTER JOIN cat2 ON cat1.kic=cat2.kic AND abs(cat1.P-cat2.P) <.1

    Parameters
    ----------
    cat1 : my catalog. Must contain `kic` and `P` fields
    cat2 : other catalog. Must contain `kic` and `P` fields

    Note
    ----
    kic and period are treated as the unique identifier for a transit


    Catalogs may contain other fields as long as they are not duplicates.

    Returns
    -------

    tcom - outer join of two catalog with columns
           kic   
           P_cat1   
           P_cat2  
           in_cat1  : convienience, tcom.in_cat1 ~tcom.P_cat1.isnull()
           in_cat2 : same
    """
    cat1 = cat1.rename(columns={'P':'P_cat1'})
    cat2 = cat2.rename(columns={'P':'P_cat2'})

    cat10 = cat1.copy()
    cat20 = cat2.copy()

    cat1 = cat1[['P_cat1','kic']]
    cat2 = cat2[['P_cat2','kic']]

    # Outer join is the union of the entries in me,cat, and both
    tcom = pd.merge(cat1,cat2,on='kic',how='outer')
    tcom = tcom[ np.abs( tcom.P_cat1 - tcom.P_cat2 ) < .1 ]
    tcom = pd.merge(cat10,tcom,on=['kic','P_cat1'],how='outer')
    tcom = pd.merge(cat20,tcom,on=['kic','P_cat2'],how='outer')
    tcom['in_cat1'] = ~tcom.P_cat1.isnull()
    tcom['in_cat2'] = ~tcom.P_cat2.isnull()

    col0 = tcom.columns
    for c in col0:
        if c.find('cat1') != -1:
            tcom = tcom.rename( columns={ c:c.replace('cat1',suffixes[0]) } )
        elif c.find('cat2') != -1:
            tcom = tcom.rename( columns={ c:c.replace('cat2',suffixes[1]) } )

    return tcom

def zerosPanel(Pb, Rpb, items='Rp1,Rp2,Rpc,P1,P2'.split(',') ):
    """
    Zeros Panel
    
    Create an empty panel.

    Parameters
    ----------
    
    Pb  : limits of the major axis (Period)
    Rpb : limits of the minor axis (Radius)
    """

    panel = pd.Panel(items=items,major_axis=Pb[:-1],minor_axis=Rpb[:-1])

    a  = np.zeros((panel.shape[1],panel.shape[2]))

    panel['Rp1'] = a + Rpb[np.newaxis,:-1]
    panel['Rp2'] = a + Rpb[np.newaxis,1:]
    panel['Rpc'] = np.sqrt(panel['Rp1'] * panel['Rp2'] )

    panel['P1'] = a + Pb[:-1,np.newaxis]
    panel['P2'] = a + Pb[1:,np.newaxis]
    panel['Pc'] = np.sqrt(panel['P1'] * panel['P2'] )
    panel.major_axis.name='P'
    panel.minor_axis.name='Rp'
    return panel

def margP(panel):
    """
    Marginalize over period
    """
    print "summing up occurrence from P = 5-50 days"
    dfRp = pd.DataFrame(index=panel.minor_axis,columns=panel.items)

    # Add the following columns over bins in P
    for k in 'fcell,fcellRaw,fcellAdd,Np'.split(','):
        dfRp[k] = panel[k].sum()


    # Add the fractional errors on ufcell in quadrature
    dfRp['ufcell1'] = np.sqrt(((panel['fuNp1']*panel['fcell'])**2).sum()) 
    dfRp['ufcell2'] = np.sqrt(((panel['fuNp2']*panel['fcell'])**2).sum()) 

    # The following are constant across P
    for k in 'Rp1,Rp2,Rpc'.split(','):
        dfRp[k] = panel[k].mean()

    return dfRp

import copy

def marg(panel,maxis):
    """
    Marginalize over a given axis
    """
    axes = panel.axes
    if maxis=='P':
        maxis=1
        kaxis=2
        firstcols = 'Rp1,Rp2,Rpc'.split(',')
    elif maxis=='Rp':
        maxis=2
        kaxis=1
        firstcols = 'P1,P2,Pc'.split(',')

    sumcols     = 'fcell,fcellRaw,fcellAdd,Np'.split(',') 
    sumquadcols = 'ufcell1,ufcell2'.split(',') 
    allcols     = firstcols + sumcols +sumquadcols
    
    df = pd.DataFrame(index=axes[kaxis] , columns=allcols )

    for k in allcols:
        arr3d = np.array(panel.ix[[k] , : ,:])
        arr3d = arr3d.swapaxes(1,maxis) # we're getting rid of first axis
        arr2d = arr3d[0]
        if sumcols.count(k)==1:
            df.ix[:,k] = np.sum(arr2d, axis=0)
        elif firstcols.count(k)==1:
            df.ix[:,k] = arr2d[0,:]
        elif sumquadcols.count(k)==1:
            df.ix[:,k] = np.sqrt( np.sum(arr2d**2,axis=0) )

    return df

def calcufcell(x):
    """

    """
    nstareff = 12e3*x['comp']
    nsamp    = 1e4
    if x['Np']==0:
        return np.zeros(nsamp)
    else:
        p = x['Np']/nstareff # probabilty for planet
        fac =   x['NpAug'] / x['Np'] / nstareff
        return np.random.binomial(nstareff,p,nsamp)*fac

def addpdf(a,b):
    """
    Add two pdfs together:

    Should add every little piece of probability 
    together. That results in too many pieces, 
    so we down-select.
    """

    assert a.size==b.size,'arrays must be of equal sizes'
    ia = random_integers(0,a.size-1,a.size)
    ib = random_integers(0,a.size-1,a.size)
    return a[ia] + b[ib]


def addPoisson(panel):
    dfshape = panel.shape[1],panel.shape[2]
    res = map(upoisson,np.array(panel['Np']).flatten())
    panel['fuNp1']    = np.array([r[0] for r in res]).reshape(dfshape)
    panel['fuNp2']    = np.array([r[1] for r in res]).reshape(dfshape)
    return panel

#######################################################################
def addFeat(df,ver=False):
    """
    Add Features

    Statistics regarding supposed transit. These features will be used
    in the automated DV step to determine if an object is a planet.

    """
    scols0 = set(df.columns) # initial list of columns

    Rstar = df.Rstar * Rsun       # Rstar [cm]
    Mstar = df.Mstar * Msun       # Mstar [g]
    P     = df.P     * sec_in_day # P [s]

    tauMa =  Rstar * P / 2 / np.pi / (df['a']*AU)

    df['tauMa']         = tauMa     / sec_in_day # Max tau given circ. orbit
    df['taur']          = df.tau0   / df.tauMa
    df['s2n_out_on_in'] = df.s2ncut / df.s2n
    df['med_on_mean']   = df.medSNR / df.s2n
    df['s2n_on_grass']  = df.s2n    / df.grass

    scols1  = set(df.columns) # final list of columns
    scols12 = scols0 ^ scols1 # 

    if ver:
        s  = \
"""\
addFeat: Added the following columns:
-------------------------------------
%s
""" % reduce(lambda x,y : x+', '+y, [str(c) for c in scols12] )
        print s

    return df

def applyCuts(df,cuts,ver=False):
    """
    Apply cuts

    One by one, test if star passed a particular cut
    bDV is true if star passed all cuts
    """

    cutkeys  = []

    for name in cuts.index:
        cut = cuts.ix[name]
        hi = float(cut['upper'])
        lo = float(cut['lower'])
        if np.isnan(hi):
            hi=np.inf
        if np.isnan(lo):
            lo=-np.inf

        cutk = 'b'+name
        cutkeys.append(cutk)
        df[cutk]=(df[name] > lo) & (df[name] < hi) 
    
    all = np.array(df[cutkeys])
    df['bDV'] = all.sum(axis=1) == len(cuts)
    return cutkeys

def found(df):
    """
    Did we find the transit?
    
    Test that the period and phase peak is the same as the input.
    
    Parameters
    ----------
    df  : DataFrame with the following columns defined
          - inj_P     : injected period
          - P         : output period
          - inj_phase : injected phase
          - t0        : output epoch (combined with P to get phase)

    Returns
    -------
    DataFrame with `phase` and `found` columns added
    """
    df['phase']     = np.mod(df.t0/df.P,1)
    dP     = np.abs( df.inj_P     - df.P     )
    dphase = np.abs( df.inj_phase - df.phase )
    dphase = np.min( np.vstack([ dphase, 1-dphase ]),axis=0 )
    dt0    = dphase*df.P
    df['found']  = (dP <.1) & (dt0 < .1) 
    return df

stellardir = '/Users/petigura/Marcy/Kepler/files/stellar/'
import sqlite3
from pandas.io import sql
def read_stellar(cat,sub=None):
    """
    Read Stellar

    Unified way to read in stellar parameters:
    
    cat : str; one of the following:
          - 'kepstellar' : Kepler stellar parameters from Exoplanet Archive
          - 'kic'        : Kepler Input Catalog
          - 'ah'         : Andrew Howard's YY corrected paramters
         
    Note
    ---- 
    Mstar is a derived parameter from Rstar and logg.

    Future Work
    -----------
    Add an attribute to DataFrame returned that keeps track of the prov.
    """

    cols = 'kic,teff,logg,prov,Rstar'.split(',')
    if cat=='kepstellar':
        cat     = '%s/keplerstellar.csv' % stellardir
        stellar = pd.read_csv(cat,skiprows=25)
        namemap = {'kepid':'kic','radius':'Rstar','prov_prim':'prov'}
        stellar = stellar.rename(columns=namemap)
    elif cat=='kic':
        cat     = '%s/kic_stellar.db' % stellardir
        con     = sqlite3.Connection(cat)
        query = 'SELECT kic,kic_teff,kic_logg,kic_radius FROM kic'
        stellar = sql.read_frame(query,con)
        namemap = {'kic_radius':'Rstar','kic_teff':'teff','kic_logg':'logg'}
        stellar = stellar.rename(columns=namemap)
        stellar = stellar.convert_objects(convert_numeric=True)
        stellar['prov'] = 'kic'
    elif cat == 'ah':
        cat     = '%s/ah_par_strip.h5' % stellardir
        store   = pd.HDFStore(cat)
        stellar = store['kic']
        namemap = {'KICID':'kic','YY_RADIUS':'Rstar','YY_LOGG':'logg',
                   'YY_TEFF':'teff'}
        stellar = stellar.rename(columns=namemap)
        stellar['prov'] = 'ah'
    else:
        print "invalid catalog"
        return None
    stellar = stellar[cols]
    stellar['Mstar'] = (stellar.Rstar*Rsun)**2 * 10**stellar.logg / G / Msun

    cat     = '%s/kic_stellar.db' % stellardir
    con     = sqlite3.Connection(cat)
    query   = 'SELECT kic,kic_kepmag FROM kic'
    mags    = sql.read_frame(query,con)
    mags    = mags.convert_objects(convert_numeric=True)
    stellar = pd.merge(stellar,mags)

    if sub is not None:
        sub = pd.DataFrame(sub,columns=['kic'])
        stellar = pd.merge(sub,stellar) 
    return stellar

def query_kic(*args):
    """
    Query KIC
   
    Connect to Kepler kic sqlite database and return PANDAS dataframe
    """
    cnx = sqlite3.connect(os.environ['KEPBASE']+'/files/db/kic_ct.db')
    if len(args)==0:
        query  = "SELECT * FROM KIC WHERE kic=8435766"
        print "enter an sqlite query, columns include:"
        print sql.read_frame(query,cnx).T        
        return None
    
    query = args[0]
    if len(args)==2:
        kic = args[1]
        kic = tuple( list(kic) )
        query += ' WHERE kic IN %s' % str( kic ) 

    df = sql.read_frame(query,cnx)
    return df

def MC(pp,res,cuts,stellar):
    pp = pp.drop('skic',axis=1)

    DV = pd.merge(pp,res,on=['outfile'],how='left')
    DV = pd.merge(DV,stellar,left_on='skic',right_on='kic')

    def wrap(DV):
        DV = addTransPars(DV)
        DV = addFeat(DV)
        DV = applyCuts(DV,cuts)
        DV = found(DV)
        DV['comp'] = DV.found & DV.bDV
        return DV

    DV['inj_Rp']    = DV['inj_p'] * DV['Rstar'] * Rsun / Rearth
    DV['P_out']     = DV['P']
    DV['phase_inp'] = DV['inj_phase']
    DV['P_inp']     = DV['inj_P']
    DV['P']         = DV['P_inp']

    DV = wrap(DV)
    return DV

# Nice logticks
xt =  [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9] + \
      [ 1, 2, 3, 4, 5, 6, 7, 8, 9] +\
      [ 10, 20, 30, 40, 50, 60, 70, 80, 90] +\
      [ 100, 200, 300, 400, 500, 600, 700, 800, 900]

sxt =  [ 0.1,  0.2,  0.3,  0.4,  0.5,  '',  '',  '',  ''] + \
       [ 1, 2, 3, 4, 5, '', '', '', ''] +\
       [ 10, 20, 30, 40, 50, '', '', '', ''] +\
       [ 100, 200, 300, 400, 500, '', '', '', '']  



def plotWhyFailDV(DV,cuts,s2n=True):
    """
    Look at the injected signals that were found, but failed DV? Why
    was that the case?
    
    s2n removes the cases that would have not passed the s2n cut.
    """

    b = DV.found & ~DV.bDV
    if s2n:
        FailDV = DV[b & DV.bs2n]
    else:
        FailDV = DV[b]

    loglog(FailDV.inj_P,FailDV.inj_Rp,'x',ms=3,mew=0.75,color='RoyalBlue',
           label='found/DV - Y/N')
    FailDV['fails'] = ''
    for cut in cuts.index:
        for i in FailDV.index:
            if ~FailDV.ix[i,'b'+cut]:
                FailDV.ix[i,'fails'] += cut+'\n'

    FailDV.apply(lambda x : text(x['inj_P'],x['inj_Rp'],x['fails'],size=4),axis=1)
    legend()
    xticks(xt,xt,rotation=45)
    
    ylim(0.5,30)
    xlim(5,400)

    xlabel('Period [days]')
    ylabel('Planet Size [Re]')

def files2bname(path):
    """
    Takes a list of files, pulls out the basename, and sticks them
    into a pandas DataFrame.
    """
    
    fL = glob.glob(path)
    df = pd.DataFrame(fL,columns=['file'])
    file2bname = lambda x : x.split('/')[-1].split('.')[0]
    df['bname'] = df.file.apply(file2bname)
    df = df.drop('file',axis=1)
    return df
    

def pmap(f,panel):
    """
    Panel Map

    Runs a function element-wise on a pandas DataFrame 
    """
    
    major_axis = panel.major_axis
    minor_axis = panel.minor_axis

    df = pd.DataFrame(columns=panel.major_axis,index=panel.minor_axis)
    for ima in major_axis:
        for imi in minor_axis:
            df.ix[imi,ima] = f(panel.major_xs(ima).ix[imi])
    return df


import copy

class TERRA():
    """
    TERRA results class
    """
    def __init__(self,pp,res,cat):
        """
        Injection and recovery obejct:

        pp   : DataFrame of Injected parameters
        res  : DataFrame of DV output
        cuts : DataFrame listing the cuts.
        """
        

        stellar       = read_stellar(cat,sub=pp)

        self.res      = res
        self.cat      = cat
        self.pp       = pp
        self.stellar  = stellar
        self.cuts     = None

        self.nlc     = self.pp.__len__()
        self.ngrid   = self.res.P.dropna().__len__()
        self.nfit    = self.res.p0.dropna().__len__()

    def __repr__(self):
        """
        String summary of injection and recovery
        """

        s = """\
Pipeline Summary
----------------
%6i Light curves submitted
%6i completed TERRA-grid
%6i completed TERRA-DV\
""" % (self.nlc , self.ngrid, self.nfit)
        
        if self.cuts is not None:
            s += self.smry()

        return s

    def subsamp_args(self,stars):
        stars = pd.DataFrame(stars,columns=['kic'])
        pp    = pd.merge(stars,self.pp)
        res   = pd.merge(stars,self.res)
        return pp,res,self.cat
            
    def smry(self):
        smry = self.cuts.copy()
        DV      = self.getDV()
        smry['pass'] = 0
        smry['only'] = 0
        s2npasscol = '+s2n>%.1f' % self.cuts.ix['s2n','lower']
        
        smry[s2npasscol ] = 0 
        for name in self.cuts.index:
            DVpass = DV['b%s' % name]
            smry.ix[name,'pass']     = DVpass.sum()
            smry.ix[name,s2npasscol] = (DVpass & DV.bs2n).sum()
            
            if name!='s2n':
                allbut = copy.copy(self.cutkeys)
                bname = 'b'+name
                allbut.remove(bname)
                allbut.remove('bs2n')
                print allbut
                bOther = array(DV[allbut]).sum(axis=1)==len(allbut)
                bOnly  = bOther & ~DV[bname] & DV['bs2n']
                smry.ix[name,'only'] = bOnly.sum()

            s = """
Cuts Summary
------------
%s

nTCE = %i
""" % (smry.to_string(),DV.bDV.sum())

        return s

    def mergeFrames(self):
        """
        Take the pp, res, and stellar dataframe and merge them 
        """

        comb = pd.merge( self.pp,self.res,on=['outfile','kic'],how='left' )
        comb = pd.merge( comb , self.stellar, on='kic' )

        Rstar = comb['Rstar'] * Rsun                   # Rstar [cm]
        Mstar = comb['Mstar'] * Msun                   # Mstar [g]
        P     = comb['P']*sec_in_day                   # P [s]
        a     = (P**2*G*Mstar / 4/ np.pi**2)**(1./3) # in cm 

        comb['a/Rstar'] = a / Rstar
        comb['a']       = a / AU
        comb['Rp']      = comb['p0'] * comb['Rstar'] * Rsun /Rearth

        return comb.drop_duplicates()
        
    def getDV(self):
        """
        Generate DV frame (usually too verbose to deal with at the top
        level).
        """
        DV = self.mergeFrames()
        DV = addFeat(DV)
        self.cutkeys = applyCuts(DV,self.cuts)
        return DV

    def setGrid(self,gridName):
        if gridName is 'terra1yr':
            self.Pb   = array([6.25,12.5,25,50.0,100,200,400])
            self.Rpb  = array([0.5, 1.0, 2,4.0,8.0,16.])        
        elif gridName is 'terra50d':
            self.Pb   = array([5,10.8,23.2,50])
            self.Rpb  = array([0.5,0.7,1.0,1.4,2,2.8,4.0,5.6,8.0,11.6,16.])        

class MC(TERRA):
    def __init__(self,pp,res,cat):
        TERRA.__init__(self,pp,res,cat)
        self.setGrid('terra50d')

    def subsamp(self,stars):
        stars = pd.DataFrame(stars,columns=['kic'])
        pp,res,cat = self.subsamp_args(stars)
        mc  = MC(pp,res,cat)
        mc.cuts = self.cuts
        return mc

    def getDV(self):
        DV = self.mergeFrames()
        DV = addFeat(DV)
        self.cutkeys = applyCuts(DV,self.cuts)
        DV['inj_Rp']  = DV['inj_p'] * DV['Rstar'] *  Rsun / Rearth
        DV = found(DV)
        DV['comp'] = DV.found & DV.bDV
        return DV

    def plotDV(self):
        plotDV( self.getDV() )
   
    def getPanel(self):
        """Return panel with completeness"""
        cPnl = zerosPanel(self.Pb,self.Rpb)
        cPnl = completeness( cPnl , self.getDV() )
        return cPnl


class TPS(TERRA):
    """
    Transiting Planet Search object
    """
    def __init__(self,pp,res,cat):
        TERRA.__init__(self,pp,res,cat)
        self.tce = None
        self.maxRpPlanet = 20 # Objects larger than 20 Re, will
                                # automatically be considered EBs

    def __add__(self,tps2):
        pp_comb  = pd.concat( [self.pp,tps2.pp] )
        res_comb = pd.concat( [self.res,tps2.res] )
        assert self.cat==tps2.cat,"two catalogs must be equal"
        tps = TPS(pp_comb,res_comb,self.cat)

        tps.cuts = self.cuts
        if self.tce is not None:
            tps.tce = pd.concat( [self.tce,tps2.tce] )

        return tps

    def subsamp(self,stars):
        stars = pd.DataFrame(stars,columns=['kic'])
        pp,res,cat = self.subsamp_args(stars)
        tps  = TPS(pp,res,cat)
        tps.tce  = pd.merge(stars,self.tce,left_on='kic',right_index=True)
        tps.cuts = self.cuts
        return tps

    def make_triage(self,path):
        """
        Make Triage Directories makes the following folders.

        path/TCE
        path/eKOI
        path/notplanet
        path/tce.txt
        path/cuts.txt
        """

        def softmkdir(path):
            try: 
                os.mkdir(path)
            except OSError:
                print path+" exists"
                pass

        def softto_csv(df,path,**kwargs):
            if ~os.path.exists(path):
                df.to_csv(path,**kwargs)                
            else:
                print path+" exists"
            
        softmkdir("%s/" %path)
        softmkdir("%s/TCE" %path)
        softmkdir("%s/eKOI"%path)

        tcepath = "%s/TCE.txt"  % path
        cutpath = "%s/cuts.csv" % path

        DV = self.getDV()
        file2skic = lambda x : x.split('/')[-1].split('.')[0]
        skic = DV[DV.bDV].outfile.apply(file2skic)

        softto_csv(skic,tcepath,index=False)
        softto_csv(self.cuts,cutpath)
        
    def read_triage(self,path):
        self.tce  = read_pngtree('%s/pngtree.txt' % path)
        ekoi = self.geteKOI()
        ekoi.index = ekoi.kic
        kicEB = ekoi[~ekoi.notplanet & (ekoi.Rp > 20)].kic
        print "%6i more FPs due to Rp > %i Re" % (len(kicEB),self.maxRpPlanet)
        self.tce.ix[kicEB,'notplanet'] = True
        self.tce.ix[kicEB,'notplanetdes'] = 'radius'
                
    def geteKOI(self):
        ekoi = self.tce[self.tce.eKOI]
        addcols = 'P,Rp,kic,a/Rstar'.split(',')
        DV = self.getDV()[addcols]
        return pd.merge(ekoi,DV,left_index=True,right_on='kic')

    def ploteKOI(self):
        loglog()        
        ekoi = self.geteKOI()
        cut = ekoi[~ekoi.notplanet]
        plot(cut.P,cut.Rp,'.',mew=0,ms=5,label='Candidate')
        cut = ekoi[ekoi.notplanet]
        plot(cut.P,cut.Rp,'x',ms=3,mew=1,label='FP')
        
        legend()
        xticks(xt,sxt)
        yticks(xt,sxt)
        xlabel('Period [days]')
        ylabel('Rp [Earth-radii]')

        xlim(5,500)
        ylim(0.5,16)

    def getHiresSum(self,obsm):
        """
        Get summary of HIRES observations
        
        Observation summary structure. 
        obsm = kbcUtils.loadKepObsm()
        """
        ekoi = self.geteKOI()
        ekoi = pd.merge(ekoi,self.stellar[['kic','kic_kepmag']])
        ekoi = pd.merge(ekoi,obsm,left_on='kic',right_index=True,how='left')
        ekoi = ekoi.fillna({'ntemp':0,'niod':0})
        return ekoi

    def plotHiresSum(self,obsm):
        """
        Produce a plot showing graphically which stars have HIRES observations:
        """
        ekoi = self.getHiresSum(obsm)
        kw = dict(ms=3,mew=1)

        # Plot the Candidates
        cut = ekoi[~ekoi.notplanet]
        plot(cut.P,cut.Rp,'s',mfc='none',mec='Tomato',**kw)
        cut = cut[cut.ntemp > 0]
        plot(cut.P,cut.Rp,'s',mfc='Tomato',mec='Tomato',**kw)

        # Plot the FPs
        cut = ekoi[ekoi.notplanet]
        plot(cut.P,cut.Rp,'s',mfc='none',mec='RoyalBlue',**kw)
        cut = cut[cut.ntemp > 0]
        plot(cut.P,cut.Rp,'s',mfc='RoyalBlue',mec='RoyalBlue',**kw)

        def tt(x):
            s = "%(kic)i\n%(kic_kepmag).1f (%(ntemp)i,%(niod)i)" % x
            text(x['P'],x['Rp'],s,size=3)

        xlabel('Period [days]')
        ylabel('Rp [Earth-radii]')

        xl  = xlim()
        ekoi[ekoi.P.between(*xl)].apply(tt ,axis=1)


class Occur():
    def __init__(self,tps,mc):
        self.tps = tps
        self.mc  = mc
        self.mc.setGrid('terra50d')

    def OccurPanel(self):
        cPnl = self.mc.getPanel()
        ekoi = self.tps.geteKOI()
        plnt = ekoi[~ekoi.notplanet]
        return occurrence(plnt,cPnl,self.tps.nlc)
    
    def occurAnn(self):
        occur = self.OccurPanel()
        occur = addpercen(occur)
        addlines(occur)

        def anntext(x):
            s = " %(Np)-2i (%(NpAug).1f)  %(fcellp).2f%%\n %(compp)i%%  " % x
            text( x['P1'] , x['Rp2'] , s, size=6,va='top') 

        occur.to_frame().apply(anntext,axis=1)

def addlines(panel):
    Pb,Rpb = getbins(panel)
    Pb,Rpb = np.array(Pb) , np.array(Rpb)
    colors = np.zeros(panel.fcell.T.shape)
    pcolor(Pb,Rpb, colors, edgecolors='LightGrey',lw=1,cmap=cm.gray_r)


def occurrence(df_tps,cPnl,nstars):
    """

    Parameters
    ----------
    Pb     : bins in period
    Rpb    : bins in planet radius
    df_tps : dataframe with the following columns
             - P 
             - Rp
             - a/Rstar
    comp   : DataFrame must match the size of the panel

    """

    panel = cPnl.copy()
    bins  = getbins(cPnl)

    def countPlanets(**kw):
        return h2d(df_tps['P'],df_tps['Rp'],bins=bins,**kw)[0]

    panel['Np']       = countPlanets(weights=ones(len(df_tps)))
    panel             = addPoisson(panel)
    panel['NpAug']    = countPlanets( weights=df_tps['a/Rstar'] )

    panel['fcellRaw'] = panel['NpAug']    / nstars
    panel['fcell']    = panel['fcellRaw'] / panel['comp']
    panel['fcellAdd'] = panel['fcell'] - panel['fcellRaw']

    panel['ufcell1'] = panel['fcell'] * panel['fuNp1']
    panel['ufcell2'] = panel['fcell'] * panel['fuNp2']

    panel['log10Rp']  = np.log10(panel['Rp2']/panel['Rp1'])
    panel['log10P']   = np.log10(panel['P2']/panel['P1'])
    panel['flogA']    = panel['fcell'] / (panel['log10Rp'] * panel['log10P'])
    return panel

def addpercen(panel):
    panel['fcellp']  = 100*panel['fcell']
    panel['compp']   = 100*panel['comp']
    panel['flogAp']  = 100*panel['flogA']
    return panel
    



def plotDV(DV):
    """
    Displays results from injection and recovery.
    """
    b = DV.found & DV.bDV
    loglog(DV[b].inj_P,DV[b].inj_Rp,'s',ms=1,mew=.5,color='RoyalBlue',
           label='found/DV - Y/Y',mec='RoyalBlue',mfc='none')

    b = DV.found & ~DV.bDV
    loglog(DV[b].inj_P,DV[b].inj_Rp,'x',ms=2.5,mew=0.75,color='RoyalBlue',
           label='found/DV - Y/N')

    b = ~DV.found & DV.bDV
    loglog(DV[b].inj_P,DV[b].inj_Rp,'x',ms=2.5,mew=0.75,color='Tomato',
           label='found/DV - N/Y')

    b = ~DV.found & ~DV.bDV
    loglog(DV[b].inj_P,DV[b].inj_Rp,'s',ms=1,mew=.5,color='Tomato',
           label='found/DV - N/N',mec='Tomato',mfc='none')

    legend()
    xticks(xt,xt,rotation=45)
    
    ylim(0.5,64)
    xlim(5,400)

    xlabel('Period [days]')
    ylabel('Planet Size [Re]')

def read_pp(file):
    pp     = pd.read_csv(file,index_col=0)
    pp     = pp.rename(columns={'skic':'kic'})
    return pp

def read_res(file,**kwargs):
    res = pd.read_csv(file,**kwargs)
    res = res.rename(columns={'skic':'kic'})
    return res

def read_pngtree(path,centroid=False):
    df = pd.read_table(path,names=['path'])
    df['dir1'] = df.path.apply(lambda x :x.split('/')[1])
    
    if not centroid:
        df['kic']  = df.path.apply(lambda x :int(x.split('/')[-1][:-7]))
    else:
        df['kic']  = df.path.apply(lambda x :int(x.split('/')[-1][4:13]))

    df.index=df.kic

    eKOI = df[df.dir1=='eKOI']
    eKOI['eKOI'] = True
    eKOI = eKOI[['eKOI']]

    notplanet = df[df.dir1=='notplanet']
    notplanet['notplanet'] = True
    notplanet['notplanetdes']  = notplanet.path.apply(lambda x: x.split('/')[2][2:])
    notplanet = notplanet[['notplanet','notplanetdes']]
    
    if not centroid:
        tce  = df[df.dir1=='TCE']
        tce['TCE'] = True
        tce = tce[['TCE']]

        tce = pd.concat([tce,eKOI,notplanet],axis=1)
        tce = tce.fillna(False)

        tce['eKOI']      = tce.eKOI.astype(bool)
        tce['notplanet'] = tce.notplanet.astype(bool)

        print """\
%6i stars designated TCE   
%6i stars designated eKOI  
%6i stars look like EBs    
""" %  (len(tce) , len(tce[tce.eKOI]) , len( tce[tce.eKOI & tce.notplanet] ))

        return tce
    else:
        ekoi = pd.concat([eKOI,notplanet],axis=1)
        ekoi = ekoi.fillna(False)

        ekoi['eKOI']      = ekoi.eKOI.astype(bool)
        ekoi['notplanet'] = ekoi.notplanet.astype(bool)
        return ekoi

def read_triage(path):
    """
    Read Triage Directory
    
    Looks for TCE.txt, eKOI.txt, notplanet.txt. If these files don't
    exist, we make them from the list of pngs in the TCE/, eKOI/, and
    notplanet/ folders
    """

    # TCEs
    tce = txtpng('triage/TCE')
    # eKOIs
    ekoi = txtpng('triage/eKOI')
    tce  = pd.merge(tce,ekoi,how='left')

    # EBs
    notplanet = txtpng('triage/notplanet')
    tce = pd.merge(tce,notplanet,how='left')
    tce = tce.fillna(False)

    tce['eKOI']      = tce.eKOI.astype(bool)
    tce['notplanet'] = tce.notplanet.astype(bool)

    print """\
%6i stars designated TCE   
%6i stars designated eKOI  
%6i stars look like EBs    
""" %  (len(tce) , len(tce[tce.eKOI]) , len( tce[tce.eKOI & tce.notplanet] ))
    return tce

def txtpng(path):
    """
    Given a path to the dircetory e.g.:
    
       ./TCE/

    look for ./TCE.txt and read and return DataFrame. If the file does
    not exist, build it out of all png files in TCE/
    """
    
    basename = path.split('/')[-1]
    pathtxt = '%s.txt' % path
    pathpngs ='%s/*.png' % path

    try:
        with open(pathtxt) as f:
            df = pd.read_table(f,sep='\s*',names=['bname'])
    except IOError:
        print """\
Constructing %s.txt
from         %s
""" % (path,pathpngs)
        df = files2bname(pathpngs)
        df.bname.to_csv(pathtxt,index=False)

    df[basename] = True
    return df
  
from astropy.io import fits
    
def read_SMEatVandy(path):
    """
    Read SME-at-Vandy fits tables
    """
    
    tab = fits.open(path)[1].data
    names = tab.dtype.names

    d = {}
    for n in names:
        d[n] = tab[n][0]
    
    return pd.DataFrame(d)
        
    
    
