
"""
Phase folded ligth curves for Tim Morton

"""
import numpy as np
from numpy import ma
import pandas as pd
import tval
import h5py
import sys
import h5plus
import config
import os
import keplerio
from matplotlib import mlab
import glob
import prepro
import terra

MORTONDIR  = os.environ['MORTONDIR']
KOIDIR     = MORTONDIR+'koilists/'

def getParJR(koi):
    """
    Get Parameters from Jason Rowe

    Given a KOI name, return the following fields to be used in the folding
    - P [days]
    - t0 epoch, properly shifted
    - tdur [days]
    - df [ppm]
    """
    jr = pd.read_csv(KOIDIR+'koi_Aug8.csv')
    jr = jr.ix[~np.isnan(jr.KepID)]
    jr = jr.ix[~np.isnan(jr.KOI)]
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
    df = pd.read_csv(KOIDIR+'KOI-list_Jan-21.csv',skiprows=3)
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

def getPar(koi,file):
    """

    """
    bname = file.split('/')[-1]
    if bname=='cumulative_2013jul26.csv':
        df = pd.read_csv(file,skiprows=76)

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
    return row


def phaseFoldKOI(row):
    """
    Phase Fold KOI

    Takes reads in a light curve from the h5 directory, stiches the
    quarters together, and folds the data on the published
    ephemeris. Must contain the following fields:

    - koi
    - skic
    - P, t0, tdur
    - pkname : where we will write out the file.

    """
    pkname = row['pkname']
    with h5plus.File(pkname,'c') as pk:
        h5plus.add_attrs(pk,row)
        try:
            h5dir = os.path.join(os.environ['PROJDIR'],'Kepler/h5files/')
            fL    = glob.glob(h5dir+'*.h5')
            raw  = pk.create_group('/raw')

            for f in fL:
                q = int(f.split('/')[-1].split('.')[0][1:])
                r = terra.getkic(f,int(row['skic']))
                if r != None:
                    r = prepro.modcols(r)
                    raw['Q%i' % q] = r

            pk.create_group('/pp')
            prepro.mask(pk)
            prepro.sQ(pk,stitch_groups='raw')

            phaseFold(pk)
            pk.attrs['status'] = 'Pass'
        except:
            status = 'Failed: Other'
            print sys.exc_info()[1]
            pk.attrs['status'] = status
        
def phaseFold(pk):
    print """
Folding %(koi)s on
------------------
P    - %(P).2f
tdur - %(tdur).2f
""" % dict(pk.attrs)
    lc = pk['/pp/mqcal'][:]
    lc['fmask'] = lc['desat'] | lc['atTwk'] | lc['isBadReg']
    lc['fmask'] = lc['fmask'] | np.isnan(lc['f'])

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

    at_Season(pk)

def at_Season(h5):
    """
    Phase-Folded and binned plot on a season by season basis
    """
    PF = h5['lcPF0'][:]
    qarr = PF['qarr']
    for season in range(4):
        try:
            bSeason = (qarr>=0) & (qarr % 4 == season)
            x = PF['tPF'][bSeason]
            y = PF['f'][bSeason]
            bw = 30. / 60. /24.
            xmi,xma = x.min(),x.max()
            nbins    = xma-xmi
            nbins = int( np.round( (xma-xmi)/bw ) )
            bins  = np.linspace(xmi,xma+bw*0.001,nbins+1 )
            tb    = 0.5*(bins[1:]+bins[:-1])
            yb = tval.bapply(x,y,bins,np.median)
            dtype = [('t',float),('fmed',float)]
            r    = np.array(zip(tb,yb),dtype=dtype )
            h5['PF_Season%i' % season] = r

        except:
            print "problem with season %i " % season
            pass
