import pandas as pd
import numpy as np
from numpy import histogram2d as h2d
from scipy.stats import poisson


comp_ness_flds = 'P,Rp,comp'.split(',')

def upoisson(Np):
    """
    Poisson Uncertanty

    Returns fractional uncertanty (asymetric)
    """
    
    x  = np.arange(100)
    cdf= poisson.cdf(x,Np)

    xi = np.linspace(0,x[-1],1000)
    yi = np.interp(xi,x,cdf)
    lo = xi[np.argmin(np.abs(yi-.159))]
    hi = xi[np.argmin(np.abs(yi-.841))]
    lo = (Np-lo) / Np
    hi = (hi-Np) / Np
    return lo,hi


def completeness(Pb,Rpb,df_mc):
    """
    Compute completeness

    df_mc  : dataframe with the following columns
             - P
             - Rp
             - comp (whether a particular simluation counts toward
               completeness)

    """    
    for k in comp_ness_flds:
        assert list(df_mc.columns).count(k) == 1,'Missing %s' % k
    
    df_comp = df_mc[ df_mc['comp'] ]
    bins = Pb,Rpb
    nPass,xe,ye = h2d(df_comp.P,df_comp.Rp, bins=bins)
    nTot,xe,ye  = h2d(df_mc.P,df_mc.Rp, bins=bins)
    comp        = nPass/nTot
    comp[np.isnan(comp)] = 0 
    return comp

def occurrence(Pb,Rpb,df_tps,comp):
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

    bins = Pb,Rpb
    nstars = 12e3
    items = 'fcell,fcellRaw,fcellAdd,Np,NpAug,comp,fuNp1,fuNp2'.split(',')

    panel = pd.Panel(items=items,major_axis=Pb[:-1],minor_axis=Rpb[:-1])
    panel2 =  binDataFrame(Pb,Rpb)
    dfshape = panel.shape[1],panel.shape[2]

    for i in panel2.items:
        panel[i]=panel2[i]

    def countPlanets(**kw):
        return h2d(df_tps['P'],df_tps['Rp'],bins=bins,**kw)[0]

    panel['Np']       = countPlanets( weights=np.ones( len(df_tps) )) 


    panel             = addPoisson(panel)
    panel['NpAug']    = countPlanets( weights=df_tps['a/Rstar'] )
    panel['comp']     = comp

    panel['fcellRaw'] = panel['NpAug']    / nstars
    panel['fcell']    = panel['fcellRaw'] / panel['comp']
    panel['fcellAdd'] = panel['fcell'] - panel['fcellRaw']

    panel['log10Rp']  = np.log10(panel['Rp2']/panel['Rp1'])
    panel['log10P']   = np.log10(panel['P2']/panel['P1'])
    panel['flogA']    = panel['fcell'] / (panel['log10Rp'] * panel['log10P'])
    return panel


def compareCatalogs(me,cat):
    """
    Compare Catalogs
    
    Determine which planets appear in both catalogs. Candidates are
    considered equal if:

    |P_me - P_cat| < 0.01 days

    I believe the SQL equivalent is:
    
    SELECT * FROM me OUTER JOIN cat ON me.kic=cat.kic AND abs(me.P-cat.P) <.1

    Parameters
    ----------
    me  : my catalog. Must contain `kic` and `P` fields
    cat : other catalog. Must contain `kic` and `P` fields

    Note
    ----
    Catalogs may contain other fields as long as they are not duplicates.

    Returns
    -------

    tcom - outer join of two catalog
    """
    me   = me.rename(columns={'P':'P_me'})
    cat  = cat.rename(columns={'P':'P_cat'})

    me0  = me.copy()
    cat0 = cat.copy()

    me   = me[['P_me','kic']]
    cat  = cat[['P_cat','kic']]

    # Outer join is the union of the entries in me,cat, and both
    tcom = pd.merge(cat,me,how='inner',on='kic')
    tcom = tcom[np.abs(tcom.P_me-tcom.P_cat) < .01] # planets in both catalogs

    tcat = pd.merge(cat,tcom,how='left',on=['kic','P_cat'])
    tcat = tcat[tcat.P_me.isnull()] # These appear only in cat

    tme  = pd.merge(me,tcom,how='left',on=['kic','P_me'])
    tme  = tme[tme.P_cat.isnull()] # These appear only in me

    # shared planets, planets in tme not in cat
    tcom = pd.concat( [tcom , tme, tcat] )

    # Join the remaining planets back on
    tcom = pd.merge(tcom,me0,how='left',on=['kic','P_me'])
    tcom = pd.merge(tcom,cat0,how='left',on=['kic','P_cat'])
    return tcom

def binDataFrame(Pb,Rpb):
    """
    """
    items='Rp1,Rp2,Rpc,P1,P2'.split(',')
    panel = pd.Panel(items=items,major_axis=Pb[:-1],minor_axis=Rpb[:-1])

    a  = np.zeros((panel.shape[1],panel.shape[2]))
    panel['Rp1'] = a + Rpb[np.newaxis,:-1]
    panel['Rp2'] = a + Rpb[np.newaxis,1:]
    panel['Rpc'] = np.sqrt(panel['Rp1'] * panel['Rp2'] )

    panel['P1'] = a + Pb[:-1,np.newaxis]
    panel['P2'] = a + Pb[1:,np.newaxis]
    panel['Pc'] = np.sqrt(panel['P1'] * panel['P2'] )
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

def margRp(panel0):
    """
    Marginalize over Rp
    """
    panel = panel0.copy()

    panel = panel0.swapaxes(copy=True)
    RpRange = panel.major_axis[panel.major_axis >=1]
    print "using the following Rps", RpRange
    panel = panel.ix[:,RpRange]
    df = pd.DataFrame(index=panel.minor_axis,columns=panel.items)

    # Add the following columns over bins in Rp
    for k in 'fcell,fcellRaw,fcellAdd,Np'.split(','):
        df[k] = panel[k].sum()

    for k in 'P1,P2,Pc,log10P'.split(','):
        df[k] = panel[k].mean()

    # Add the following errors in quadrature
    df['ufcell1'] = np.sqrt(((panel['fuNp1']*panel['fcell'])**2).sum())
    df['ufcell2'] = np.sqrt(((panel['fuNp2']*panel['fcell'])**2).sum())
    return df

def addPoisson(panel):
    dfshape = panel.shape[1],panel.shape[2]
    res = map(upoisson,np.array(panel['Np']).flatten())
    panel['fuNp1']    = np.array([r[0] for r in res]).reshape(dfshape)
    panel['fuNp2']    = np.array([r[1] for r in res]).reshape(dfshape)
    return panel



#######################################################################


# Commonly used functions for dealing with cuts
### Mstar, 
G = 6.672e-8 # [cm3 g^-1 s^-2]
Rsun = 6.955e10 # cm
Rearth = 6.3781e8 # cm
Msun = 1.9891e33 # [g]
AU   = 1.49597871e13 # [cm]
sec_in_day = 86400


def addCuts(df):
    """
    Add the following DV statistics (ratios of other DV statistics)
    """
    scols0 = set(df.columns) # initial list of columns

    Rstar = df.Rstar * Rsun # Rstar [cm]
    Mstar = df.Mstar * Msun # Mstar [g]
    P = df.P_out*sec_in_day # P [s]

    a = (P**2*G*Mstar / 4/ np.pi**2)**(1./3) # in cm 
    df['a/R*'] = a/Rstar
    P = df.P_out*sec_in_day # P [s]
    tauMa =  Rstar*P/2/np.pi/a
    df['tauMa']         = tauMa / sec_in_day # Max tau given circ. orbit
    df['taur']          = df.tau0 / df.tauMa
    df['s2n_out_on_in'] = df.s2ncut / df.s2n
    df['med_on_mean']   = df.medSNR / df.s2n
    df['Rp']            = np.sqrt(df.df0)*df.Rstar*109.04
    df['phase_out']     = np.mod(df.t0/df.P_out,1)
    scols1 = set(df.columns) # final list of columns
    scols12 = scols0 ^ scols1 # 
    
    print "addCuts: Added the following columns"
    print "-"*80
    for c in scols12:
        print c

    return df

def applyCuts(df,cuts):
    """
    Apply cuts

    One by one, test if star passed a particular cut
    bDV is true if star passed all cuts
    """

    cutkeys  = []
    nPass    = []
    nPassS2N = []

    for name in cuts['name']:
        cut = cuts.ix[np.where(cuts.name==name)[0]]
        hi = float(cut['upper'])
        lo = float(cut['lower'])
        if np.isnan(hi):
            hi=np.inf
        if np.isnan(lo):
            lo=-np.inf

        cutk = 'b'+name
        cutkeys.append(cutk)
        df[cutk]=(df[name] > lo) & (df[name] < hi) 

        nPass.append( len(df[df[cutk]]) ) 
        nPassS2N.append( len(df[df[cutk] & df['bs2n'] ]) ) 

    df['bDV'] = df[cutkeys].T.sum()==len(cutkeys)
    summary = cuts.copy()
    summary['nPass'] = nPass
    summary['nPassS2N'] = nPassS2N

    print "applyCuts: summary"
    print "-"*80
    print summary.to_string(index=False)
    print "%i stars passed all cuts" % len(df[df.bDV])
    return df

def found(df):
    """
    Did we find the transit?
    
    Test that the period and phase peak is the same as the input
    
    Parameters
    ----------
    df  : DataFrame with the following columns defined
          - P_inp
          - P_out
          - phase_inp
          - phase_out

    Returns
    -------
    found column
    """

    dP     = np.abs( df.P_inp     - df.P_out     )
    dphase = np.abs( df.phase_inp - df.phase_out )
    dphase = np.min( np.vstack([ dphase, 1-dphase ]),axis=0 )
    dt0    = dphase*df.P_out
    df['found'] = (dP <.1) & (dt0 < .1) 
    return df


import config

b12k = pd.read_csv('../TERRA2/b12k.csv')
b12k['skic'] = b12k.kic.astype('|S10').str.pad(9).str.replace(' ','0')
b12k = b12k['kic,skic,a1,a2,a3,a4'.split(',')]

store = pd.HDFStore('../files/ah_par_strip.h5')
ah_par = store['kic']
ah_par['kic'] = ah_par['KICID']

keepcols = [c for c in ah_par.columns if (c.find('YY_')!=-1) or (c.find('kic')!=-1)]
ah_par   = ah_par[keepcols]
ah_par['Rstar'] = ah_par.YY_RADIUS
ah_par['Mstar'] = ah_par.YY_MSTAR
ah_par         = ah_par[['Mstar','Rstar','kic']]


def MC(pp,res,cuts):
    DV = pd.merge(pp,res,on=['outfile','skic'],how='left')
    print DV.columns

    DV =  pd.merge(DV,ah_par,left_on='skic',right_on='kic')
    DV['df0'] = DV['p0']**2
    DV['Re']  = DV['inj_p'] * DV['Rstar'] * 109.

    def wrap(DV):
        DV = addCuts(DV)
        DV = applyCuts(DV,cuts)
        DV = found(DV)
        DV['pass'] = DV.found & DV.bDV
        return DV

    DV['P_out']     = DV['P']
    DV['phase_inp'] = DV['inj_phase']
    #DV['phase_inp'] = DV['inj_phase']
    DV['P_inp'] = DV['inj_P']
    DV['P'] = DV['P_inp']
    #DV['bname'] = DV['sid']

    print cuts

    DV = wrap(DV)
    return DV
