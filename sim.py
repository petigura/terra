import atpy
import platform
import os
import stat
from numpy import ma
import glob
import numpy as np
import copy

import join
import qalg
import tfind
import ebls
import tval
import keptoy
import keplerio

def grid(t,f,Psmp=0.25):
    PG0 = ebls.grid( t.ptp() , 0.5, Pmin=50.0, Psmp=Psmp)
    res = tfind.tfindpro(t,f,PG0)
    tRES = qalg.dl2tab([res])
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
    return tVAL





def kwUnique(kwL,key):
    """
    Confirm that all instances of keyword are unique.  Return that unique value
    """
    valL = np.unique( np.array( [ kw[key] for kw in kwL ]  ) )
    assert valL.size == 1, '%s must be unique' % key
    return valL[0]

def convEpoch(epoch0,P,t0):
    """
    Convert Epoch

    There are a couple of conventions for epochs.  keptoy.genEmpLC
    injects signals at the input epoch.  The transit finder zeroes out
    the starting time.  The output epoch will not be the same.
    """
    
    return np.remainder(epoch0 + t0,P)


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

def simReduce(files):
    """
    Reduce output of a simulation.
    """

    if files[0].count('tVAL') == 1:
        dL = map(bfitVAL,files)
    else:
        dL = map(bfitRES,files)

    tRED =  qalg.dl2tab(dL)
    tRED.set_primary_key('seed')
    return tRED

def bfitRES(file):
    """
    Returns the best fit parameters from tRES.
    """
    tRES = atpy.Table(file,type='fits')
    
    idMa = np.argmax(tRES.s2n[0])
    d = dict(
        oP     = tRES.PG[0][idMa],
        oepoch = tRES.epoch[0][idMa],
        odf    = tRES.df[0][idMa],
        os2n   = tRES.s2n[0][idMa],
        otwd   = tRES.twd[0][idMa],
        seed  = tRES.keywords['SEED']
        )
    return d

def bfitVAL(file):
    """
    Returns the best fit parameters from tVAL.
    """
    tVAL = atpy.Table(file,type='fits')

    s2n = ma.masked_invalid(tVAL.s2n)
    idMa = np.argmax(s2n)

    d = dict(
        oP     = tVAL.P[idMa],
        oepoch = tVAL.epoch[idMa],
        odf    = tVAL.df[idMa],
        os2n   = tVAL.s2n[idMa],
        otwd   = tVAL.tdur[idMa],
        seed   = tVAL.keywords['SEED']
        )
    return d

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

def inject(tLCbase,tPAR):
    """
    Inject transit signal into lightcurve.
    """
    assert tPAR.data.size == 1
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

def PARRES(tPAR0,tRED0):
    """

    """
    tPAR = copy.deepcopy(tPAR0)
    tRED = copy.deepcopy(tRED0)

    tPAR.set_primary_key('seed')
    tRED.set_primary_key('seed')
    tRED = join.join(tPAR,tRED)

    # Convert epochs back into input
    tRED.data['oepoch'] = np.remainder(tRED.oepoch,tRED.P)
    addbg(tRED)
    addFlag(tRED)
    return tRED


def addbg(t):    
    bg = qalg.bg(t.P,t.oP,t.epoch,t.oepoch)
    keplerio.update_column(t,'bg',bg)    
    return t

def diagFail(t):
    """
    Why did a particular run fail?
    """
    kepdir = os.environ['KEPDIR']
#    tLC  = atpy.Table(t.keywords['LCFILE'],type='fits')
#    pknown = qalg.tab2dl(t)[0]
#    f = keptoy.genEmpLC(pknown , tLC.t,tLC.f  )
#    dM,bM,aM,DfDt,f0 = tfind.MF(f,20)
#    Pcad = round(pknown['P']/keptoy.lc)
#    res = tfind.ep(tLC.t,dM,Pcad)
#    win = res['win']
#    # Likely explaination for failure : window function.
#    bwin = ~(win[np.floor(t.epoch[0]/keptoy.lc)]).astype(bool)
    
    return balias,bwin


def addFlag(t):
    """
    Adds a string description as to why the run failed'
    """
    balias = map(qalg.alias,t.P,t.oP)
    keplerio.update_column(t,'balias', )
#    keplerio.update_column(t,'bwin',bwinL)
    return t
