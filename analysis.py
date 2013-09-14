import pandas as pd
import numpy as np
from numpy import histogram2d as h2d
import scipy.stats
import glob
import os
from matplotlib.pylab import *

from cStringIO import StringIO
from config import G,Rsun,Rearth,Msun,AU,sec_in_day
from astropy.io import fits
import copy

koilistdir = '/Users/petigura/Marcy/Kepler/files/koilists/'
def loadKOI(cat):
    if cat=='CB':
        cat = pd.read_table('%s/koi_Burke_20dec2012.tab' % koilistdir,sep='|')
        namemap = {' keplerId ':'kic',' koi_disposition ':'disp',
                   ' koi_period ':'P','KOI       ':'koi',' koi_prad ':'Rp'}
        cat = cat.rename(columns=namemap)[namemap.values()]
        cat['disp'] = cat.disp.apply(lambda x : x.strip())
    elif cat=='Q12':
        namemap = {'kepid':'kic','kepoi_name':'koi',
                   'koi_pdisposition':'disp',
                   'koi_period':'P','koi_depth':'df',
                   'koi_duration':'tdur','koi_prad':'Rp'}
        path = koilistdir+'cumul_2013sep13.csv'
        cat = pd.read_csv(path,skiprows=74).rename(columns=namemap)[namemap.values()]        
    return cat

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

    bins = getpanelbins(panel)
    
    df_comp       = df_mc[ df_mc['comp'] ]
    nPass , xe,ye = h2d(df_comp.inj_P , df_comp.inj_Rp , bins=bins)
    nTot ,  xe,ye = h2d(df_mc.inj_P   , df_mc.inj_Rp   , bins=bins)         

    panel['nPass'] = nPass
    panel['nTot']  = nTot
    panel['comp'] = nPass/nTot
    panel['comp'] = panel.comp.fillna(0)
    return panel


def compareCatalogs(cat1,cat2,suffixes=['cat1','cat2'],Pthresh=0.1):
    """
    Compare Catalogs
    
    Determine which planets appear in both catalogs. Candidates are
    considered equal if:

    |P_cat1 - P_cat2| < 0.1 days

    I believe the SQL equivalent is:
    
    SELECT * FROM cat1 OUTER JOIN cat2 ON cat1.kic=cat2.kic AND abs(cat1.P-cat2.P) <.1

    Parameters
    ----------
    cat1 : my catalog. Must contain `kic` and `P` fields
    cat2 : other catalog. Must contain `kic` and `P` fields
    Pthresh : periods must be within 0.1 days to agree

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
    tcom = tcom[ np.abs( tcom.P_cat1 - tcom.P_cat2 ) < Pthresh ]
    tcom = pd.merge(cat10,tcom,on=['kic','P_cat1'],how='outer')
    tcom = pd.merge(cat20,tcom,on=['kic','P_cat2'],how='outer')
    tcom['in_cat1'] = ~tcom.P_cat1.isnull()
    tcom['in_cat2'] = ~tcom.P_cat2.isnull()
    tcom['kic'] = tcom.kic.astype(int)

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

def addpcols(df0,lables):
    """
    Given a list of columns, add in a percentage column
    fcell -> pfcell
    """
    df = df0.copy()
    for l in lables:
        pl = 'p%s' % l 
        df[pl] = 100 * df[l]
    return df

def marg(panel,maxis):
    """
    Marginalize over a given axis
    """

    axes = panel.axes
    if maxis=='P':
        maxis=1
        firstcols = 'Rp1,Rp2,Rpc'.split(',')
    elif maxis=='Rp':
        maxis=2
        firstcols = 'P1,P2,Pc'.split(',')
        panel = panel.swapaxes(1,2)

    sumcols     = 'fcell,fcellRaw,fcellAdd,Np,NpAdd'.split(',') 
    sumquadcols = 'ufcell1,ufcell2'.split(',') 
    allcols     = firstcols + sumcols + sumquadcols

    iaxes = [1,2]      # Keep the axis that we're not moving
    iaxes.remove(maxis)
    kaxis = iaxes[0]

    df = pd.DataFrame(index=axes[kaxis] , columns=allcols )

    for k in allcols:
        table = panel.loc[k]
        if sumcols.count(k)==1:
            df.ix[:,k] = table.sum()
        elif firstcols.count(k)==1:
            df.ix[:,k] = table.iloc[0]
        elif sumquadcols.count(k)==1:
            df.ix[:,k] = np.sqrt( (table**2).sum() )

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
    def __init__(self,pp,res,stellar):
        """
        Base class for TERRA results (both inj/rec and TPS)

        Parameters
        ----------
        pp      : DataFrame of Inputs to TERRA
        res     : DataFrame of DV output
        stellar : DataFrame of stellar properies
        """

        self.res      = res
        self.pp       = pp
        self.stellar  = stellar

        self.cuts     = None
        self.nlc     = self.pp.__len__()
        self.ngrid   = self.res.P.dropna().__len__()
        self.nfit    = self.res.p0.dropna().__len__()
        self.Pb      = None
        self.Rpb     = None

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

    def subsamp(self,stars):
        """
        Return a new TERRA instance with only only the subset of stars
        """

        args0 = (self.pp,self.res,self.stellar)
        args1 = []
        for a in args0:
            a.index = a.kic
            args1.append( a.ix[stars] )
        args1 = tuple(args1)

        return TERRA(*args1)

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
        Pb,Rpb = gridDict[gridName]
        self.Pb = Pb
        self.Rpb = Rpb

gridDict = {
    'terra1yr':(
        [6.25,12.5,25,50.0,100,200,400], # Pb
        [0.5, 1.0, 2,4.0,8.0,16.]        # Rpb 
        ),
    'terra1yr-fine':(
        [6.25, 8.84, 12.5, 17.7, 25, 35.4, 50.0, 
         70.7, 100, 141, 200, 283, 400],
        [0.5, 0.71, 1.0, 1.41, 2, 2.82, 4.0, 5.66, 8.0, 11.6, 16.]
        ),
    'terra50d':(
        [5,10.8,23.2,50],
        [0.5,0.7,1.0,1.4,2,2.8,4.0,5.6,8.0,11.6,16.]
        )
    }

for k in gridDict:
    Pb,Rpb = gridDict[k]
    gridDict[k] = (np.array(Pb),np.array(Rpb) )


class MC(TERRA):
    def __init__(self,pp,res,stellar):
        TERRA.__init__(self,pp,res,stellar)

    def subsamp(self,stars):
        mc = TERRA.subsamp(self,stars) 
        mc.__class__ = MC
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

    def plotDV(self,**kwargs):
        plotDV( self.getDV(),**kwargs )
        labelPRp()
   
    def getPanel(self):
        """Return panel with completeness"""
        cPnl = zerosPanel(self.Pb,self.Rpb)
        cPnl = completeness( cPnl , self.getDV() )
        return cPnl

    def plotCompPanel(self):
        """
        Lay down a colored checker borad showing the completeness.
        """
        cPnl    = self.getPanel()
        colors  = array( floor(cPnl['comp']*10)/10 +.05 ).T

        pcolor(self.Pb, self.Rpb, colors, cmap='RdBu',
               vmin=-.3,vmax=1.3,edgecolors='white',lw=1)

        f = lambda x : text(x['P1'],x['Rp1'],x['scomp'],size='x-small')
        cPnl['scomp'] = (cPnl['comp'] * 100).astype(int).astype(str) + '%' 
        cPnl.to_frame().apply(f,axis=1)

maxRpPlanet = 20 # Objects larger than 20 Re, will

class TPS(TERRA):
    """
    Transiting Planet Search object
    """
    def __init__(self,pp,res,stellar):
        TERRA.__init__(self,pp,res,stellar)
        self.tce = None
                                # automatically be considered EBs

    def __add__(self,tps2):
        pp_comb  = pd.concat( [self.pp,tps2.pp] )
        res_comb = pd.concat( [self.res,tps2.res] )
        assert self.stellar is tps2.stellar,"Use the same stellar parameters"
        tps = TPS(pp_comb,res_comb,self.stellar)

        tps.cuts = self.cuts
        if self.tce is not None:
            tps.tce = pd.concat( [self.tce,tps2.tce] )

        return tps

    def subsamp(self,stars):
        tps = TERRA.subsamp(self,stars) 
        tps.__class__ = TPS
        tps.cuts = self.cuts        
        tps.tce  = self.tce.ix[stars].dropna(how='all')

        return tps

    def getTCE(self):
        """
        Return a list of TCEs for manual vetting.
        """
        DV = self.getDV()
        tce = DV[DV.bDV]

        keys = []
        for k in self.cuts.index:
            keys += [k,'b'+k] 


        file2skic = lambda x : x.split('/')[-1].split('.')[0]
        tce['skic'] = tce.outfile.apply(file2skic)
        tce.index = tce.kic

        keys += ['bDV','skic']
        tce = tce[keys]
        return tce

    def read_triage(self,path):
        """
        Attach list of TCEs.
        - Read in eKOI list
        - Read in my designations
        - Read in Kepler team centroiding designations
        """
        self.tce = read_pngtree('%s/me/pngtree.txt' % path)
        tce = self.tce[['eKOI']]
        tce['dsg'] = ''
        dsg = tce['dsg']
        
        ekoi      = self.geteKOI()
        dsg.ix[ekoi[ekoi.Rp > maxRpPlanet].index] = 'Rp'

        # My own FP assessment
        myFP = self.tce[self.tce.eKOI]['notplanetdes']
        for field in 'eclipse,vardepth,ttv,Vshape'.split(','):
            dsg = add_triage(dsg,myFP,field)

        centFP = read_pngtree('%s/centroid/pngtree.txt' % path,centroid=True)
        centFP = centFP['notplanetdes']
        for field in 'offset-star,offset-no-star'.split(','):
            dsg = add_triage(dsg,centFP,field)

        tce['dsg'] = dsg
        tce['myFP']   = myFP
        tce['centFP'] = centFP
        self.tce = tce

    def getFPtab(self):
        ekoi = self.geteKOI()
        
        FPcode = """
Rp             R
eclipse        SE
vardepth       VD
ttv            TTV
Vshape         V
offset-no-star C
offset-star    C
"""
        FPcode = StringIO(FPcode)
        FPcode = pd.read_table(FPcode,sep='\s*',squeeze=True,index_col=0)
        dsg = ekoi.dsg
        dsg = dsg.replace( dict(FPcode) )
        dsg = dsg.replace('','plnt')
        return dsg
        
    def geteKOI(self):
        ekoi    = self.tce[self.tce.eKOI]
        addcols = 'P,Rp,kic,a/Rstar'.split(',')
        DV      = self.getDV()[addcols]
        ekoi    = pd.merge(ekoi,DV,left_index=True,right_on='kic')
        ekoi.index = ekoi.kic
        return ekoi

    def ploteKOI(self):
        loglog()        
        ekoi = self.geteKOI()
        ekoi['dsg'] = self.getFPtab()
        cut = ekoi[ekoi.dsg=='plnt']
        plot(cut.P,cut.Rp,'.',mew=0,ms=5,label='Candidate')
        cut = ekoi[ekoi.dsg!='plnt']
        plot(cut.P,cut.Rp,'x',ms=3,mew=1,label='FP')
        
        legend()
        xticks(xt,sxt)
        yticks(xt,sxt)
        labelPRp()

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

        labelPRp()

        xl  = xlim()
        ekoi[ekoi.P.between(*xl)].apply(tt ,axis=1)

def add_triage(dsg0,sel,field):
    """
    Add triage string.
    
    dsg : Series containing the current designations or empty strings
    
    Loops over all the indecies in the sel series. If `field` is present in
    `sel` and still '' in dsg, fill in the empty string
    """
    dsg = dsg0.copy()
    for i in sel.index:
        if (dsg.ix[i] == '') & (sel.ix[i]==field):
            dsg.ix[i] = field
    return dsg


class Occur():
    def __init__(self,tps,mc):
        self.tps = tps
        self.mc  = mc

    def OccurPanel(self):
        cPnl = self.mc.getPanel()
        ekoi = self.tps.geteKOI()
        ekoi['dsg'] = self.tps.getFPtab()
        plnt = ekoi[ekoi.dsg=='plnt']
        occur = occurrence(plnt,cPnl,self.tps.nlc)
        occur['pcomp']  = occur['comp']  * 100
        occur['pfcell'] = occur['fcell'] * 100
        occur['pflogA'] = occur['flogA'] * 100
        return occur
    
    def occurAnn(self):
        occur = self.OccurPanel()
        addlines(occur)

        def anntext(x):
            s = " %(Np)-2i (%(NpAug).1f)  %(fcellp).2f%%\n %(compp)i%%  " % x
            text( x['P1'] , x['Rp2'] , s, size=6,va='top') 

        occur.to_frame().apply(anntext,axis=1)

def plotOccur2D(occur,Pb,Rpb,plnt,compThresh=0.25):
    """
    Draw 2D occurrence distribution
    """

    plot(plnt.P,plnt.Rp,'s',color='Tomato',mew=0,mec='Tomato',ms=3)


    def f(x):
        offset = 1.01
        s = """\
%(Np)i (%(NpAug).1f)
%(pcomp)i%%""" % x
        text(x['P1']*offset,x['Rp1']*offset,s,size='x-small',ha='left')

        s = """\
%(pfcell).2f%%
%(pflogA).2f%%""" % x
        text(x['P2']/offset,x['Rp1']*offset,s,size='x-small',ha='right')

    occur.to_frame().apply(f,axis=1)

    step  = 0.01

    occur['colors'] = occur['fcell']
    colors = ma.masked_array( occur['colors'] , occur['comp'] < compThresh ).T
    maxco = np.ceil( colors.max()/step ) * step

    pcolor(Pb,Rpb,colors,cmap='YlGn',vmin=0,vmax=maxco*1.5,
           edgecolors='LightGrey',lw=1)
    boundaries=linspace(0,maxco,maxco/step + 1)
    cb = colorbar(drawedges=False,boundaries=boundaries,shrink=.5)
    cb.set_label('Planet Occurrence',size='small')

def getbins(df,name):
    bins = list(df[name+'1']) + list(df[name+'2'])
    bins = list(set(bins))
    bins.sort()
    return bins

def getpanelbins(panel):
    lua = lambda x : list(np.unique(np.array(x)))
    Rpb = lua(panel.Rp1) + [ lua(panel.Rp2)[-1] ]
    Pb  = lua(panel.P1)  + [ lua(panel.P2)[-1]  ]
    bins = Pb,Rpb
    return bins

def plotOccur1D(dfMarg,name):
    """
    Plot one dimensional occurrence distribution

    name : 'P' or 'Rp'
    """
    sbinc = name+'c'
    binc = list(dfMarg[sbinc])
    bins = getbins(dfMarg,name)

    hist([binc]*2, bins=bins, 
         weights=[dfMarg.fcellRaw,dfMarg.fcellAdd],
         color=['DarkGrey','Tomato'],
         label=['Raw occurrence','Correction for\n missed planets'],
         histtype='barstacked',rwidth=.98,edgecolor='w')
    
    yerr = vstack([dfMarg.ufcell1,dfMarg.ufcell2])
    errorbar(dfMarg[sbinc],dfMarg.fcell,yerr=yerr,fmt='o',ms=4,color='k')
    
    ax = gca()
    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    def f(x):
        s = "%(pfcell).1f %%" % x
        y = x['fcell'] + x['ufcell2']+0.002
        text(x[sbinc],y,s,ha='center')
    dfMarg.apply(f,axis=1)


    def f2(x):
        kw = dict(size='x-small',ha='center',va='center',color='w')

        y  = x['fcellRaw'] / 2 
        s  = '%(Np)i' % x
        text(x[sbinc],y,s,**kw)

        y  = x['fcell'] - x['fcellAdd'] / 2 
        s  = '%(NpAdd).1f' % x
        text(x[sbinc],y,s,**kw)
    dfMarg.apply(f2,axis=1)        


def addlines(panel):
    Pb,Rpb = getpanelbins(panel)
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

    pnl = cPnl.copy()
    bins = getpanelbins(cPnl)

    def countPlanets(**kw):
        return h2d(df_tps['P'],df_tps['Rp'],bins=bins,**kw)[0]

    pnl['Np']       = countPlanets(weights=ones(len(df_tps)))
    pnl             = addPoisson(pnl)
    pnl['NpAug']    = countPlanets( weights=df_tps['a/Rstar'] )

    pnl['fcellRaw'] = pnl['NpAug']    / nstars
    pnl['fcell']    = pnl['fcellRaw'] / pnl['comp']
    pnl['fcellAdd'] = pnl['fcell'] - pnl['fcellRaw']
    pnl['NpAdd']    = pnl['Np'] * ( 1/pnl['comp'] - 1 )

    pnl['ufcell1'] = pnl['fcell'] * pnl['fuNp1']
    pnl['ufcell2'] = pnl['fcell'] * pnl['fuNp2']

    pnl['log10Rp']  = np.log10(pnl['Rp2']/pnl['Rp1'])
    pnl['log10P']   = np.log10(pnl['P2']/pnl['P1'])
    pnl['flogA']    = pnl['fcell'] / (pnl['log10Rp'] * pnl['log10P'])
    return pnl

    
def plotDV(DV,cases=2):
    """
    Displays results from injection and recovery.
    
    Parameters
    ----------
    DV - Data Validation frame. Must contain
         - found
         - bDV
         - inj_P
         - inj_Rp

    cases - Number of different outcomes to plot
    """
    def cplot(b,*args,**kwargs):
        plot(DV[b].inj_P,DV[b].inj_Rp,*args,**kwargs)

    if cases==4:
        b  = DV.found & DV.bDV
        cplot(b,'s',ms=1,mew=.5,color='RoyalBlue',
              label='found/DV - Y/Y',mec='RoyalBlue',mfc='none')  

        b = DV.found & ~DV.bDV
        cplot(b,'x',ms=2.5,mew=0.75,color='RoyalBlue',
               label='found/DV - Y/N')

        b = ~DV.found & DV.bDV
        cplot(b,'x',ms=2.5,mew=0.75,color='Tomato',
               label='found/DV - N/Y')

        b = ~DV.found & ~DV.bDV
        cplot(b,'s',ms=1,mew=.5,color='Tomato',
               label='found/DV - N/N',mec='Tomato',mfc='none')

    elif cases==2:
        b = DV.found & DV.bDV
        cplot(b,'s',ms=2,mfc='RoyalBlue',mew=0,lw=0,mec='RoyalBlue')
        b = ~(DV.found & DV.bDV)
        cplot(b,'s',ms=2,mfc='Tomato',mew=0,lw=0,mec='Tomato')

def read_pp(file):
    pp     = pd.read_csv(file,index_col=0)
    pp     = pp.rename(columns={'skic':'kic'})
    return pp

def read_res(file,**kwargs):
    res = pd.read_csv(file,**kwargs)
    res = res.rename(columns={'skic':'kic'})
    res = res[res.kic.notnull()]
    res['kic'] = res.kic.astype(int)
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
    
    path2notplanetdes  = lambda x: x.split('/')[2][2:]
    notplanet['notplanetdes'] = notplanet.path.apply(path2notplanetdes)
    notplanet = notplanet[['notplanet','notplanetdes']]
    
    if not centroid:
        tce  = df[df.dir1=='TCE']
        tce['TCE'] = True
        tce = tce[['TCE']]

        tce = pd.concat([tce,eKOI,notplanet],axis=1)
        tce = tce.fillna(False)

        tce['eKOI']      = tce.eKOI.astype(bool)
        tce['notplanet'] = tce.notplanet.astype(bool)

        d = {'ntce' : tce.TCE,
             'neKOI': tce.eKOI,
             'nEB'  : tce.notplanet & (tce.notplanetdes!='ttv'),
             'nTTV' : tce.notplanet & (tce.notplanetdes=='ttv')}

        for k in d.keys():
            d[k] = d[k].sum()
        
        print """\
%(ntce)6i stars designated TCE   
%(neKOI)6i stars designated eKOI  
""" %  d

        return tce
    else:
        ekoi = pd.concat([eKOI,notplanet],axis=1)
        ekoi = ekoi.fillna(False)

        ekoi['eKOI']      = ekoi.eKOI.astype(bool)
        ekoi['notplanet'] = ekoi.notplanet.astype(bool)
        return ekoi












def labelPRp():
    xlabel('Orbital Period [days]')
    ylabel('Planet Size [Earth-radii]')

# Nice logticks
xt =  [ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9] + \
      [ 1, 2, 3, 4, 5, 6, 7, 8, 9] +\
      [ 10, 20, 30, 40, 50, 60, 70, 80, 90] +\
      [ 100, 200, 300, 400, 500, 600, 700, 800, 900]

sxt =  [ 0.1,  0.2,  0.3,  0.4,  0.5,  '',  '',  '',  ''] + \
       [ 1, 2, 3, 4, 5, '', '', '', ''] +\
       [ 10, 20, 30, 40, 50, '', '', '', ''] +\
       [ 100, 200, 300, 400, 500, '', '', '', '']  

def getlogticks(xl):
    """
    Returns log ticks for [0.1,0.2,0.3,0.4,0.4,0.5,1,2,3,4,5]
    """
    
    xl = log10(np.array(xl))
    xl[0] = floor(xl[0])
    xl[1] = ceil(xl[1])
    xl = xl.astype(int)

    ticks  = []
    sticks = []
    for i in range(*xl):
        for j in range(1,10):
            t = 10**i * j

            # Take care of the float rounding
            if i < 0:
                s = '%s' % float('%.1g' % t)
            else:
                s = '%s' % t 

            if j>5:
                s = ''

            ticks += [t]
            sticks += [s]
    return ticks,sticks

def logticks(axis):
    if axis=='both':
        xticks(*getlogticks( xlim() ) )
        yticks(*getlogticks( ylim() ) ) 
    elif axis=='x':
        xticks(*getlogticks( xlim() ) )
    elif axis=='y':
        yticks(*getlogticks( ylim() ) ) 
