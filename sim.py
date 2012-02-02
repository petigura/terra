import atpy
import platform
import os
import stat
from numpy import ma
import glob
import numpy as np
import copy

import qalg
import tfind
import ebls
import tval
import keptoy
import keplerio

def grid(tLC,Psmp=0.25):
    PG0 = ebls.grid( tLC.t.ptp() , 0.5, Pmin=50.0, Psmp=Psmp)
    res = tfind.tfindpro(tLC.t,tLC.f,PG0)
    tRES = qalg.dl2tab([res])
    tRES.comments = "Table with the simulation results"
    tRES.table_name = "RES"
    tRES.keywords = tLC.keywords
    return tRES

def val(tLC,tRES,nCheck=50,ver=True):
    dL = tval.parGuess(qalg.tab2dl(tRES)[0],nCheck=nCheck)
    resL = tval.fitcandW(tLC.t,tLC.f,dL,ver=ver)

    # Alias Lodgic.
    # Check the following periods for aliases.
    resLHigh = [r for r in resL if  r['s2n'] > 5]
    resLLow  = [r for r in resL if  r['s2n'] < 5] 	
    
    resLHigh = tval.aliasW(tLC.t,tLC.f,resLHigh)
            
    # Combine the high and low S/N 

    resL = resLHigh + resLLow
    tVAL = qalg.dl2tab(resL)
    tVAL.keywords = tRES.keywords
    return tVAL


def iPoP(tval):
    """
    """
    s2n = ma.masked_invalid(tval.s2n)
    iMax = s2n.argmax()
    t = tval.rows( [iMax] )
    return t

def diagFail(t):
    """
    Why did a particular run fail?
    """
    kepdir = os.environ['KEPDIR']
    Palias = t.P[0] * np.array([0.5,2])

    # Likely explaination for failure : alias function.
    balias = (abs( t.oP[0] / Palias - 1) < 1e-3).any()
    tLC  = atpy.Table(t.keywords['LCFILE'],type='fits')
    pknown = qalg.tab2dl(t)[0]
    f = keptoy.genEmpLC(pknown , tLC.t,tLC.f  )
    dM,bM,aM,DfDt,f0 = tfind.MF(f,20)
    Pcad = round(pknown['P']/keptoy.lc)
    res = tfind.ep(dM,Pcad)
    win = res['win']
    # Likely explaination for failure : window function.
    bwin = ~(win[np.floor(t.epoch[0]/keptoy.lc)]).astype(bool)
    
    return balias,bwin


def addFlag(t):
    """
    Adds a string description as to why the run failed'
    """

    baliasL,bwinL = [],[]
    for i in range(len(t.data)):
        balias,bwin = diagFail(t.rows([i]))
        baliasL.append(balias)
        bwinL.append(bwin)
    keplerio.update_column(t,'balias',baliasL)
    keplerio.update_column(t,'bwin',bwinL)
    return t


def kwUnique(kwL,key):
    """
    Confirm that all instances of keyword are unique.  Return that unique value
    """
    valL = np.unique( np.array( [ kw[key] for kw in kwL ]  ) )
    assert valL.size == 1, '%s must be unique' % key
    return valL[0]

def resRed(files,PARfile=None,LCfile=None):
    """
    Result Reduce.

    Read in all the tVAL results and augment tPAR with the results.

    Parameters
    ----------
    files : A list of tVAL to reduce.
    """

    # Build table containing input parameters.
    seedL,parfileL=[],[]

    tVALL = [atpy.Table(f,type='fits') for f in files]

    if (PARfile == None) | (LCfile == None):
        kwL = [t.keywords for t in tVALL ]

    if PARfile == None:
        PARfile = kwUnique(kwL,'PARFILE')

    if PARfile == None:
        LCfile = kwUnique(kwL,'LCFILE')

    tPAR = atpy.Table(PARfile,type='fits')

    for f in files:
        bname = os.path.basename(f)
        bname = os.path.splitext(bname)[0]
        seed = int(bname.split('tVAL')[-1])
        seedL.append(seed)

    tRED = tPAR.rows(seedL)
    tRED.keywords = tVALL[0].keywords
    
    ttemp = iPoP( atpy.Table(files[0],type='fits') )
    for f in files[1:]:
        tnew = iPoP( atpy.Table(f,type='fits') )
        ttemp.append( tnew )

    col = ['s2n','P','epoch','tdur','df'] # columns to attach
    for c in col:
        keplerio.update_column(tRED,'o'+c,ttemp[c])    

    addbg(tRED)
    addFlag(tRED)
    return tRED

def convEpoch(epoch0,P,t0):
    """
    Convert Epoch

    There are a couple of conventions for epochs.  keptoy.genEmpLC
    injects signals at the input epoch.  The transit finder zeroes out
    the starting time.  The output epoch will not be the same.
    """
    
    return np.remainder(epoch0 + t0,P)

def addbg(t):    
    bg = ( abs(t.P - t.oP)/t.P < 0.001 ) & ( abs(t.epoch - t.oepoch) < 0.1 )
    keplerio.update_column(t,'bg',bg)    
    return t

def getkw(file):
    """
    Open all the files.  Return list of KW dictionaries.
    """    
    t = atpy.Table(file)
    return t.keywords

def getkwW(files,view=None):
    if view==None:
        return map(getkw,files)
    else:
        return view.map(getkw,files)

def quickRED(RESfiles,view=None):
    """
    Quick look at the periodogram.  Do not incorporate tVAL information.
    """

    kwL = getkwW(RESfiles,view=view)
    PARfileL = [kw['PARFILE'] for kw in kwL]
    PARfile = np.unique(PARfileL)
    LCfileL = [kw['LCFILE'] for kw in kwL]
    LCfile = np.unique(LCfileL)
    assert PARfile.size == 1, "PARfile must be unique"
    assert PARfile.size == 1, "LCfile must be unique"

    PARfile = PARfile[0]
    LCfile = LCfile[0]

    tRED = atpy.Table(PARfile,type='fits')
    tRED.keywords['PARFILE'] = PARfile
    tRED.keywords['LCFILE'] = LCfile
    
    col = ['P','epoch'] # columns to attach
    ocol = ['o'+c for c in col]
    for o in ocol:
        tRED.add_empty_column(o,np.float)
    
    for RESfile in RESfiles:
        tRES = atpy.Table(RESfile,type='fits')
        seed = name2seed(RESfile)

        dL = tval.parGuess( qalg.tab2dl(tRES)[0] )
        d = dL[0]
        irow = np.where(tRED.seed == seed)[0]

        for c,o in zip(col,ocol):
            tRED[o][irow] = d[c]

    addbg(tRED)
    addFlag(tRED)
    return tRED

def name2seed(file):
    """
    Convert the name of a file to a seed value.
    """
    bname = os.path.basename(file)
    bname = os.path.splitext(bname)[0]
    return int(bname[-4:])
    
def addVALkw(files):
    """
    Add input information to the tVAL files
    """

    tVALL = [atpy.Table(f,type='fits') for f in files]
    kwL = [t.keywords for t in tVALL ]

    PARfile = kwUnique(kwL,'PARFILE')
    tPAR = atpy.Table(PARfile,type='fits')

    for t,f in zip(tVALL,files):
        seed = name2seed(f)
        tROW = tPAR.where(tPAR.seed == seed)
        assert tROW.data.size == 1, 'Seed must be unique'
        col = ['P','epoch','df','tdur','seed','tbase']
        for c in col:
            t.keywords[c] = tROW.data[c][0]
        t.write(f,overwrite=True,type='fits')


def inject(tLCbase,tPAR,seed):
    """
    Inject transit signal into lightcurve.
    """
    tPAR = tPAR.where(tPAR.seed == seed)
    d0 = qalg.tab2dl(tPAR)[0]    

    assert len(tPAR.data) == 1, "Seed must be unique" 
    tLC = map(keplerio.nQ,tLCbase)
    inj = lambda t : tinject(t,d0)
    tLC = map(inj,tLC)
    tLCraw = atpy.TableSet(tLC)
    tLCraw.keywords = d0

    return tLCraw

def tinject(t0,d0):
    """
    Table inject
    """    
    t = copy.deepcopy(t0)
    f = keptoy.genEmpLC(d0,t.TIME,t.f)
    keplerio.update_column(t,'f',f)
    return t
