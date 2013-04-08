import pandas
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

    panel = pandas.Panel(items=items,major_axis=Pb[:-1],minor_axis=Rpb[:-1])
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
    tcom = pandas.merge(cat,me,how='inner',on='kic')
    tcom = tcom[np.abs(tcom.P_me-tcom.P_cat) < .01] # planets in both catalogs

    tcat = pandas.merge(cat,tcom,how='left',on=['kic','P_cat'])
    tcat = tcat[tcat.P_me.isnull()] # These appear only in cat

    tme  = pandas.merge(me,tcom,how='left',on=['kic','P_me'])
    tme  = tme[tme.P_cat.isnull()] # These appear only in me

    # shared planets, planets in tme not in cat
    tcom = pandas.concat( [tcom , tme, tcat] )

    # Join the remaining planets back on
    tcom = pandas.merge(tcom,me0,how='left',on=['kic','P_me'])
    tcom = pandas.merge(tcom,cat0,how='left',on=['kic','P_cat'])
    return tcom

def binDataFrame(Pb,Rpb):
    """
    """
    items='Rp1,Rp2,Rpc,P1,P2'.split(',')
    panel = pandas.Panel(items=items,major_axis=Pb[:-1],minor_axis=Rpb[:-1])

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
    dfRp = pandas.DataFrame(index=panel.minor_axis,columns=panel.items)

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
    df = pandas.DataFrame(index=panel.minor_axis,columns=panel.items)

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
