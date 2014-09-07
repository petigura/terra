"""
Module for performing photometry of K2 data.


"""
import h5py
from numpy import ma
from matplotlib.pylab import *
import pandas as pd
import prepro
import os
from config import k2_dir,path_phot
from astropy.io import fits
from scipy import ndimage as nd

def imshow2(im,**kwargs):
    extent = None#(0,im.shape[0],0,im.shape[1])

    if kwargs.has_key('cmap')==False:
        kwargs['cmap'] = cm.gray 

    imshow(im,interpolation='nearest',origin='lower',
           extent=extent,**kwargs)

def plot_med_star(name,stretch='none'):
    ts,cube = read_pix(path_pix,name)
    fcube = cube['FLUX']
    fcube = flat_cube(fcube)
    med_image = np.median(fcube,axis=0)

    if stretch=='arcsinh':
        imshow2(arcsinh(med_image))
    else:
        imshow2(med_image)
        
c0_start_cad = 114

def get_comb(f,name):
    if hasattr(name,'__iter__') is False:
        name = [name]

    nname = len(name)
    with h5py.File(f,'r') as h5:
        try:
            h5db = pd.DataFrame( h5['name'][:].astype(int),columns=['name']) 
        except:
            h5db = pd.DataFrame( h5['epic'][:].astype(int),columns=['name']) 

        h5db['h5idx'] = np.arange(len(h5db))
        h5db['indb'] = True

        name = pd.DataFrame( name,columns=['name']) 
        name['inname'] = True
        name['nameidx'] = np.arange(nname)

        comb = pd.merge(name,h5db,how='left')
        missingname = comb[comb['indb'].isnull()].name

        assert comb['indb'].sum()==len(name), "missing %s" % missingname

        comb = comb.sort('h5idx')
    return comb


def read_k2_fits(f):
    with fits.open(f) as hduL:
        # Image cube. At every time step, one image. (440 x 50 x 50)
        fcube = 'RAW_CNTS FLUX FLUX_ERR FLUX_BKG FLUX_BKG_ERR COSMIC_RAYS'.split()
        cube = rec.fromarrays([hduL[1].data[f] for f in fcube],names=fcube)

        # Time series. At every time step, one scalar. (440)
        fts = 'TIME TIMECORR CADENCENO QUALITY POS_CORR1 POS_CORR2'.split()
        ts = rec.fromarrays([hduL[1].data[f] for f in fts],names=fts)
        aper = hduL[2].data

        head0 = dict(hduL[0].header)
        head1 = dict(hduL[1].header)
        head2 = dict(hduL[2].header)

        return ts[c0_start_cad:],cube[c0_start_cad:],aper,head0,head1,head2

def read_pix(f,name):
    """
    Read in K2 pixel data from h5 repo.

    name : List of star names to read in

    """
    basename = f.split('/')[-1] 
    with h5py.File(f) as h5:
        comb = get_comb(f,name)
        cube = h5['cube'][comb.h5idx,c0_start_cad:]
        ts = h5['ts'][comb.h5idx,c0_start_cad:]

        if len(comb) > 1:
            if ~(comb.sort('nameidx').nameidx==comb.nameidx).all():
                print "For faster reads, read in order"
                cube = cube[comb.nameidx]
                ts = ts[comb.nameidx]

        return ts,cube

def read_phot(f,name):
    """
    Read in K2 photometry from h5 directory
    """
    with h5py.File(f,'r') as h5:
        comb = get_comb(f,name)
        lc = h5['dt'][comb.h5idx,:]
    return lc

def read_cal(f,name):
    """
    Read in K2 photometry from h5 directory
    """
    with h5py.File(f,'r') as h5:
        comb = get_comb(f,name)
        lc = h5['cal'][comb.h5idx,:]
    return lc


def SAP_FLUX(i):
    image = cube['FLUX'][i,20:40,20:40]
    image -= np.median(image)   

    sources = daofind(image, fwhm=1.0, threshold=10*bkg_sigma)   

    positions = zip(sources['xcen'], sources['ycen'])   
    radius = 8.
    apertures = ('circular', radius)   
    fluxtable, aux_dict = aperture_photometry(image, positions, apertures)

    return brightest[1]

#from astropy.stats import median_absolute_deviation as mad
#from photutils import daofind
#bkg_sigma = 1.48 * mad(image)   
#from photutils import aperture_photometry

from astropy.stats import median_absolute_deviation as mad
from photutils import daofind,aperture_photometry
import numpy as np

def get_pos(cube,plot_diag=False):
    """
    """

    # Do a median through the data cube to get a high SNR starting image
    image = np.median(cube,axis=0)

    x,y = np.mgrid[:image.shape[0],:image.shape[1]]
    image_clip = image.copy()

#    bin =  (15 <= x) & (x <= 35) & (15 <= y) & (y <= 35) 
#    image_clip[~bin] = 0 


    # Perform median subtraction
    bkg_sigma = 1.48 * mad(image)   
    sources = daofind(image_clip, fwhm=1.0, threshold=100*bkg_sigma)   
    brightest = sources[sources['flux'].argmax()]
    pos = ( brightest['xcen'] , brightest['ycen'] )

    if plot_diag:
        c = Circle(pos,radius=8,fc='none',ec='red')
        gca().add_artist(c)
        imshow2(image_clip)

    return pos

def flat_cube(cube):
    med_image = np.median(np.median(cube,axis=1),axis=1)
    return cube - med_image[:,np.newaxis,np.newaxis]

def circular_photometry(ts,cube0,aper,plot_diag=False):
    """
    Circular Photometry

    Parameters
    ----------

    Returns
    -------
    ts : Time Series. Record array with the following keys
         - t
         - TIMECORR
         - cad
         - QUALITY
         - f 
         - fmask
    """
    cube = cube0.copy()

    cube['FLUX'] = flat_cube(cube['FLUX'])

    pos = nd.center_of_mass(aper==3)
    pos = (pos[1],pos[0])
    radius = 8.

    apertures = ('circular', radius)   

    nobs = cube.shape[0]
    mask = cube['FLUX'].sum(axis=1).sum(axis=1)==0
    f = ma.masked_array(np.zeros(nobs),mask)
    ferr = np.zeros(nobs)

    # Brightest sources
    for i in range(nobs):
        try:
            image = cube[i]
            fluxtable, aux_dict = aperture_photometry(image['FLUX'], pos, 
                                                      apertures)
            f.data[i] = fluxtable['aperture_sum']
        except:
            f.mask[i] = True

    ts = pd.DataFrame(ts)
    ts['f'] = f.data
    ts['fmask'] = f.mask
    ts['fraw'] = ts['f'].copy()
    ts = ts.rename(columns={'TIME':'t','CADENCENO':'cad'})
    ts = ts.drop('POS_CORR1 POS_CORR2'.split(),axis=1)
    ts['f'] = ts.f / ts.f.median() - 1

    # Normalize segment
    ts = np.array(pd.DataFrame(ts).to_records(index=False))

    if plot_diag:
        clf()
        fimagemed = np.median(cube['FLUX'],axis=0)
        imshow2( arcsinh(fimagemed) ) 
        imshow2(aper,cmap=cm.hot,alpha=0.2,vmax=10)
        c = Circle(pos,radius=8,fc='none',ec='red')
        gca().add_artist(c)

    return ts

def r2fm(r,field):
    """
    Convienence Function. Make masked array from r['f'] and r['fmask']
    """
    return ma.masked_array(r[field],r['fmask'])

from scipy.optimize import fmin
import stellar

def phot_vs_kepmag(plot_diag=False):
    with h5py.File(path_phot,'r') as h5:
        fraw = h5['dt']['fraw']
        epic = h5['epic'][:]

    fmed = np.median(fraw,axis=1)
    b = fmed > 1 
    epic = epic[b]

    df0 = stellar.read_cat()
    df0.index = df0.epic
    df = df0.copy()

    df = df.ix[epic]
    df['logfmed'] = log10(fmed[b])

    # Fit a line (robust) to log(flux) and kepmag
    p0 = [-1,14]
    obj = lambda p : np.sum(abs(df.logfmed - polyval(p,df.kepmag)))
    p1 = fmin(obj,p0,disp=0)

    df['logfmed_resid'] = df.logfmed - polyval(p1,df.kepmag)
    if plot_diag:
        plot(df.kepmag,df.logfmed,'.')
        kepmagi = linspace(9,20,100)
        plot(kepmagi,polyval(p1,kepmagi),lw=2)
        setp(gca(),xlabel='Kepmag',ylabel='Flux')

    return pd.concat([df0,df['logfmed logfmed_resid'.split()]],axis=1)

def Ceng2C0(lc0):
    """
    Simple script that turns the engineering data into C0 lenght data
    """
    lc = lc0.copy()

    tbaseC0 = 75
    tbase = lc['t'].ptp()
    lc = prepro.rdt(lc)

    # Detrend light curves to remove some of the jumps between segments
    lc['f'] = lc['fdt'] 

    nrepeat = int(np.ceil(tbaseC0/tbase))
    lcC0 = np.hstack([lc]*nrepeat)
    for i in range(nrepeat):
        s = slice(lc.size*i, lc.size*(i+1)) 
        lcC0['t'][s]+=tbase*i
        lcC0['cad'][s]+=lc['cad'].ptp()*i
    return lcC0


import glob
from pdplus import LittleEndian as LE

def read_crossfield(epic):
    pathstar = 'photometry/Ceng_pixdecor/%i_loc*.fits' % epic
    path = glob.glob(pathstar)
    
    if len(path)==0:
        print "no results for %s" % pathstar
        return None

    with fits.open(path[0]) as hduL:
        ian = pd.DataFrame(LE(hduL[1].data))

    ian = ian['cad cleanFlux noThrusterFiring'.split()]
    ian['noThrusterFiring'] = ian.noThrusterFiring.astype(bool)

    lc = read_cal('Ceng.cal.h5',60017809)
    lc = pd.DataFrame(lc)
    lc = pd.merge(lc,ian,how='left',on='cad')
    lc['fmask'] = np.isnan(lc['cleanFlux']) | (lc['noThrusterFiring']==False)
    lc['fdt'] = lc['cleanFlux']
    lc['fdt'] /= median(lc['fdt'])
    lc['fdt'] -= 1
    lc = lc.drop('fcal ftnd'.split(),axis=1)
    return np.array(lc.to_records(index=False))

