"""
Phase folded ligth curves for Tim Morton
"""
from numpy import *
import pandas
import tval
import h5py
import sys
import h5plus
import config

MORTONDIR    = '/global/scratch/sd/petigura/Morton/'

#jr.to_csv(MORTONDIR+'parfile.csv',index=False)
#with open(MORTONDIR+'koi.txt','w') as f:
#    f.writelines(["%s\n" % t for t in jr.koi])

def getParJR(koi):
    """
    Get Parameters from Jason Rowe

    Given a KOI name, return the following fields to be used in the folding
    - P [days]
    - t0 epoch, properly shifted
    - tdur [days]
    - df [ppm]
    """
    jr = pandas.read_csv('/global/homes/p/petigura/Kepler/files/koi_Aug8.csv')
    jr = jr.ix[~isnan(jr.KepID)]
    jr = jr.ix[~isnan(jr.KOI)]
    jr['skic'] = ["%09d" % i for i in jr.KepID]
    jr['koi']  = ["%.2f"  % i for i in jr.KOI]

    jr['t0'] = jr['T0'] + 67
    jr['P']  = jr['Period']
    jr['df'] = jr['Tdepth']
    jr['tdur']  = jr['Tdur'] / 24.
    jr = jr[['skic','koi','P','t0','df','tdur']]
    jr['name'] = jr.koi
    jr.index = jr['koi']

    jr['pkname'] = MORTONDIR+"pk/%(name)s.h5" % row
    jr['lcname'] = MORTONDIR+"lc/%(skic)s.h5" % row

    return jr.ix[koi]     


def getParCB(koi):
    """
    Get Parameters from Chris Burke
    
    Same as getParJR
    """
    df = pandas.read_csv(MORTONDIR+'KOI-list_Jan-21.csv',skiprows=3)
    df['skic'] = ["%09d" % i for i in df.kepid]
    df['koi']  = df.kepoi_name
    
    # Note that there is no epoch offset here, counteract the hack downstream
    df['t0'] = df.koi_time0bk + config.lc 

    df['P']  = df.koi_period
    df['tdur']  = df.koi_duration / 24.

    df['df'] = df.koi_depth
    df = df[['skic','koi','P','t0','tdur','df']]
    df['name'] = df.koi
    df.index = df['koi']
    row  = dict(df.ix[koi])
    row['pkname'] = MORTONDIR+"pk/%(name)s.h5" % row
    row['lcname'] = MORTONDIR+"lc/%(skic)s.h5" % row
    return row


def phaseFoldKOI(row):
    """
    Contains the following fields
    - koi
    - skic
    - P, t0, tdur
    """
    pkname = row['pkname']
    lcname = row['lcname']
    print "reading lc from " + lcname
    with h5py.File(lcname,'r') as lc, h5plus.File(pkname,'c') as pk:
        h5plus.add_attrs(pk,row)
        try:
            phaseFold(lc,pk)
            pk.attrs['status'] = 'Pass'
        except:
            status = 'Failed: Other'
            print status
            pk.attrs['status'] = status
        
def phaseFold(lc,pk):
    print """
Folding %(koi)s on
------------------
P    - %(P).2f
tdur - %(tdur).2f
""" % dict(pk.attrs)

    lc = lc['mqcal'][:]
    lc['fmask'] = lc['desat'] | lc['atTwk'] | lc['isBadReg']
    lc['fmask'] = lc['fmask'] | isnan(lc['f'])
    pk['mqcal'] = lc

    cpad    = 0.5 # Default fitting parameters
    cfrac   = 3
    LDT_deg = 3
    nCont   = 4

    fitWidth = lambda : 2*pk.attrs['tdur']*(cpad+cfrac)

    if fitWidth() > pk.attrs['P']:
        print "morton: Fitting region too long, shortening"
        cpad    = 0.5
        cfrac   = 1
        LDT_deg = 1

    # Expected number of points in continuum region
    if cfrac*pk.attrs['tdur'] / config.lc < nCont:
        print "morton: Short continuum region"
        cpad    = 0.5
        cfrac   = 3
        LDT_deg = 1
        nCont   = 2

    status = ''
    if fitWidth() > pk.attrs['P']:
        status = 'Failed: Pulse duty cycle too big'
        print status
        pk.attrs['status'] = status
        return        

    if cfrac*pk.attrs['tdur'] / config.lc < nCont:
        status = "Failed: Too few points in continuum"
        print status 
        pk.attrs['status'] = status
        return

    pk.attrs['cpad']    = cpad
    pk.attrs['cfrac']   = cfrac
    pk.attrs['nCont']   = nCont
    pk.attrs['LDT_deg'] = LDT_deg

    for ph in [0,180]:
        tval.at_phaseFold(pk,ph)
        tval.at_binPhaseFold(pk,ph,10)
        tval.at_binPhaseFold(pk,ph,30)    
