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
import sqlite3

import join
import qalg
import tfind
import ebls
import tval
import keptoy
import keplerio
from matplotlib import mlab
from config import *
import os
import h5py

kicdb = os.environ['KEPBASE']+'files/KIC.db'

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

def simReduce(ds,fds,type='grid'):
    """
    Reduce output of a simulation.

    ds   : A (Nrun,nP) record array
    files: A (Nrun) array of the file names

    type : Are we dealing with periodograms or validation files?
    """

    if type is 'grid':
        dL = map(bfitRES,ds[:],fds[:])
#        dL = []
#        for i in range(ds.shape[0]):
#            dL.append( bfitRES(ds[i],fds[i]) ) 
        name = 'tredg'
    elif type is 'val':
        dL = map(bfitVAL,files)
        name = 'tredv'
    else:
        print "type not understood"
        raise ValueError

    dL = np.hstack(dL)
    tRED =  qalg.rec2tab(dL)
    tRED.table_name = name
    return tRED

def tredSave(pardb,tRED):
    """
    Join the output of the tRED table with the par database
    """

    name = tRED.table_name
    tRED.write('sqlite',pardb,overwrite=True)

    con = sqlite3.connect(pardb)
    cur = con.cursor()

    cmd = """
ATTACH "%(kicdbpath)s" as kicdb;

CREATE TABLE temp AS 
SELECT * from %(name)s
JOIN par 
ON %(name)s.seed=par.seed 
JOIN kicdb.q6 
ON par.kic=kicdb.q6.id;

DROP TABLE %(name)s;

ALTER TABLE temp RENAME TO %(name)s;
""" % {'kicdbpath':kicdb,'name':name}

    cur.executescript(cmd)
    con.close()


def bfitRES(tres,file):
    """
    Returns the best fit parameters from tRES.

    Parameters
    ----------
    tres : record
    """

    idMa = np.argmax(tres['s2n'])
    oP     = tres['PG'][idMa]
    oepoch = tres['epoch'][idMa]
    odf    = tres['fom'][idMa]
    os2n   = tres['s2n'][idMa]
    otwd   = tres['twd'][idMa]
    seed   = int(file.split('_')[-1].split('.grid.')[0])

    names = ['seed','oP','oepoch','os2n','otwd','odf']
    types = [int]+[float]*5
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

def tinject(t0,d0,fluxField='f'):
    """
    Table inject
    
    Inject signal into fluxField 
    """    
    t = copy.deepcopy(t0)
    f = keptoy.genEmpLC(d0,t.t,t[fluxField])
    keplerio.update_column(t,fluxField,f)
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
    fL   = [h5py.File(f) for f in lcfiles]
    lcfilesKIC  = np.unique([f['LC'].attrs['KEPLERID'][0] for f in fL])
    tredKIC     = np.unique(tRED.KIC)
    assert (lcfilesKIC==tredKIC).all(),'tRED.KIC and tL.KIC must agree'

    # Add empty colums
    keplerio.update_column(tRED,'good',np.empty(tRED.data.size))
    keplerio.update_column(tRED,'harm',np.empty(tRED.data.size))
    keplerio.update_column(tRED,'win',np.empty(tRED.data.size))

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
        LC1d = [f['LC1d'] for f in fL if f['LC'].attrs['KEPLERID'][0]==KIC][0]
        t        = LC1d['t']
        f        = LC1d['f']
        isStep   = LC1d['isStep']
        fmask    = LC1d['fmask']

        dM  = tfind.mtd(t,f,isStep,fmask,12)
        fbwin = lambda P,epoch : bwin(dM.mask,t[0],P,epoch)
        win   = map(fbwin,tr['P'],tr['epoch'])
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

