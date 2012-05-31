"""
Simulation

This module implements the transit search pipeline.  

Note - EAP 5/22/2012 

Data is stored .fits tables and accessed via
atpy.  At some point, I'd like to migrate over to .h5 and h5py because
it's a more standard format.  This module deals with the unpacking of
the tables only passing numpy arrays to `tval` and `tfind` modules.
"""
import atpy
import os
from numpy import ma
import numpy as np
import copy

import join
import qalg
import tfind
import ebls
import tval
import keptoy
import keplerio
from matplotlib import mlab
from config import *


def grid(t,fm,isStep,**kwargs):
    PG0   = ebls.grid( t.ptp() , 0.5, **kwargs)

    # Chunck it up.
    PG0L = [ PG0[i:i+blockSize] for i in range(0, PG0.size, blockSize) ]
    def func(PG):
        rec2d = tfind.tdpep(t,fm,isStep,PG)
        rec   = tfind.tdmarg(rec2d)
        return rec
    recL = map(func,PG0L)
    rec  = np.hstack(recL)

    tRES  = qalg.rec2tab(rec)
    return tRES

def val(tLC,tRES,ver=True):
    t  = tLC.t
    fm = ma.masked_array(tLC.fdt-tLC.fcbv,mask=tLC.fmask)

    tres = tRES.data    
    tres = mlab.rec_append_fields(tres,['P','tdur','df'], \
        [tres['PG'],tres['twd']*keptoy.lc,tres['fom']])

    tresfit,tresharm = tval.val(t,fm,tres)
    tVAL = qalg.rec2tab(tresharm)

    return tVAL

def simReduce(files,type='grid'):
    """
    Reduce output of a simulation.
    """

    if type is 'grid':
        dL = map(bfitRES,files)
    elif type is 'val':
        dL = map(bfitVAL,files)
    else:
        print "type not understood"
        raise ValueError

    dL = np.hstack(dL)
    tRED =  qalg.rec2tab(dL)

    return tRED

def bfitRES(file):
    """
    Returns the best fit parameters from tRES.
    """
    tRES = atpy.Table(file,type='fits')
    
    idMa = np.argmax(tRES.s2n)
    oP     = tRES.PG[idMa]
    oepoch = tRES.epoch[idMa]
    seed   = int(file.split('_')[-1].split('.grid.')[0])
    odf    = tRES.fom[idMa]
    os2n   = tRES.s2n[idMa]
    otwd   = tRES.twd[idMa]

    names = ['seed','oP','oepoch','os2n','otwd','odf']
    types = [int] + [float]*5
    dtype = zip(names,types)
    res = np.array([(seed,oP,oepoch,os2n,otwd,odf)],dtype=dtype)

    return res

def bfitVAL(file):
    """
    Returns the best fit parameters from tVAL.
    """
    tVAL = atpy.Table(file,type='fits')

    s2n = ma.masked_invalid(tVAL.s2n)
    idMa = np.argmax(s2n)

    seed   = int(file.split('_')[-1].split('.val.')[0])
    oP     = tVAL.P[idMa],
    oepoch = tVAL.epoch[idMa],
    odf    = tVAL.df[idMa],
    os2n   = tVAL.s2n[idMa],
    otwd   = tVAL.tdur[idMa],

    names = ['seed','oP','oepoch','odf','os2n','otwd']
    types = [int] + [float]*5
    dtype = zip(names,types)

    res = np.array([(seed,oP[0],oepoch[0],odf[0],os2n[0],otwd[0])],dtype=dtype)

    return res

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
    f = keptoy.genEmpLC(d0,t.t,t.f)
    keplerio.update_column(t,'f',f)
    return t

def addbg(t):    
    bg = qalg.bg(t.P,t.oP,t.epoch,t.oepoch)
    keplerio.update_column(t,'bg',bg)    
    return t

def addFlags(tRED,lcfiles):
    """
    Add Flags.

    The following boolean comments to the tRED table
    - good : True if oP,oepoch are consistent with P,epoch
    - harm : True if oP is a harmonic of P
    - win  : True if P,oP was consistent with window function.

    Parameters
    ----------
    tRED   : atpy results table.  must contain
             oP,oepoch,P,epoch,KIC columns
    lcfiles: list of light curve files.  Must be one for every unique KIC
    """
    
    load = lambda f : atpy.Table(f,type='fits')
    tL = map(load,lcfiles)

    lcfilesKIC  = np.unique([t.keywords['KEPLERID'] for t in tL])
    tredKIC     = np.unique(tRED.KIC)
    assert (lcfilesKIC==tredKIC).all(),'tRED.KIC and tL.KIC must agree'

    # Add empty colums
    tRED.add_empty_column('good',np.int)
    tRED.add_empty_column('harm',np.int)
    tRED.add_empty_column('win',np.int)

    tred = tRED.data
    for KIC in tredKIC:
        mKIC = tred['KIC'] == KIC        
        tr = tred[mKIC]

        # Is oP,oepoch consistent with P,epoch?
        good = map(bg,tr['P'],tr['oP'],tr['epoch'],tr['oepoch'])
        tr['good'] = good

        # Is oP a harmonic of P
        harm = map(bharm,tr['P'],tr['oP'])
        tr['harm'] = harm
        
        # Would the transit be accessible after masking?
        tLC = [t for t in tL if t.keywords['KEPLERID']==KIC][0]
        dM  = ma.masked_array(tLC.dM6,tLC.dM6mask)        
        fbwin = lambda P,epoch : bwin(dM.mask,tLC.t[0],P,epoch)
        win   = map(fbwin,tr['oP'],tr['epoch'])
        tr['win'] = win
        tred[mKIC] = tr

    return tRED


def bg(P,oP,epoch,oepoch):
    """
    True if oP,oepoch is consistent with P,epoch
    """
    return ( abs(P - oP)/P < Plim ) & ( abs(epoch - oepoch) < epochlim )

def bharm(P,oP):
    """
    True if oP is a harmonic of P
    """    
    Pharm = P * harmL
    return (abs( oP / Pharm - 1) < Plim).any()

def bwin(fmask,t0,P,epoch):
    """
    Boolean Window
    
    Did a particular combination of (P,epoch) get excluded based on the window?

    Parameters
    ----------
    fmask : boolean array corresponding to the window function.
    P     : period of supposed transit
    epoch : epoch of supposed transit

    Returns
    -------
    True if P,epoch are accessible after masking

    """

    dM     = ma.masked_array( np.ones(fmask.size) , fmask )
    Pcad   = round(P/keptoy.lc)
    res    = tfind.ep(t0,dM,Pcad)

    # Find the first epoch after time = 0 .
    epochG = np.remainder(res['epoch'],P)
    idnn   = np.argmin( abs(epochG - epoch) )
    bwin   = res['win'][idnn].astype(bool)
    return bwin

