"""
Module for performing photometry of K2 data.


"""
import h5py
from numpy import ma
from matplotlib.pylab import *
import pandas as pd
import prepro


def imshow2(im,**kwargs):
    extent = (0,im.shape[0],0,im.shape[1])
    imshow(im,interpolation='nearest',origin='lower',
           cmap=cm.gray,extent=extent,**kwargs)

c0_start_cad = 114

def read_k2_fits(f):
    """
    Read in K2 pixel data from fits file
    """

    hduL = fits.open(f)

    # Image cube. At every time step, one image. (440 x 50 x 50)
    fcube = 'RAW_CNTS FLUX FLUX_ERR FLUX_BKG FLUX_BKG_ERR COSMIC_RAYS'.split()
    cube = rec.fromarrays([hduL[1].data[f] for f in fcube],names=fcube)

    # Time series. At every time step, one scalar. (440)
    fts = 'TIME TIMECORR CADENCENO QUALITY POS_CORR1 POS_CORR2'.split()
    ts = rec.fromarrays([hduL[1].data[f] for f in fts],names=fts)
    return ts,cube
    

def get_comb(f,epic):
    if hasattr(epic,'__iter__') is False:
        epic = [epic]

    nepic = len(epic)
    with h5py.File(f,'r') as h5:
        h5db = pd.DataFrame( h5['epic'][:].astype(int),columns=['epic']) 
        h5db['h5idx'] = np.arange(len(h5db))
        h5db['indb'] = True

        epic = pd.DataFrame( epic,columns=['epic']) 
        epic['inepic'] = True
        epic['epicidx'] = np.arange(nepic)

        comb = pd.merge(epic,h5db,how='left')
        missingepic = comb[comb['indb'].isnull()].epic

        assert comb['indb'].sum()==len(epic), "missing %s" % missingepic

        comb = comb.sort('h5idx')
    return comb



def read_pix(f,epic):
    """
    Read in K2 pixel data from h5 repo.

    epic : List of star names to read in

    """
    basename = f.split('/')[-1] 
    with h5py.File(f) as h5:
        comb = get_comb(f,epic)
        cube = h5['cube'][comb.h5idx,c0_start_cad:]
        ts = h5['ts'][comb.h5idx,c0_start_cad:]

        if len(comb) > 1:
            if ~(comb.sort('epicidx').epicidx==comb.epicidx).all():
                print "For faster reads, read in order"
                cube = cube[comb.epicidx]
                ts = ts[comb.epicidx]

        return ts,cube

def read_phot(f,epic):
    """
    Read in K2 photometry from h5 directory
    """
    with h5py.File(f,'r') as h5:
        comb = get_comb(f,epic)
        lc = h5['dt'][comb.h5idx,:]
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

def circular_photometry(ts,cube0,plot_diag=False):
    """
    Circular Photometry

    Parameters
    ----------
    epic : K2 epic ID

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

    pos = (25,25)
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
 
#            fluxtable, aux_dict = aperture_photometry(image['FLUX_ERR'], pos, 
#                                                      apertures)
#            ferr[i] = np.fluxtable['aperture_sum']
#
        except:
            f.mask[i] = True


    ts = pd.DataFrame(ts)
    ts['f'] = f.data
    ts['fmask'] = f.mask
    ts = ts.rename(columns={'TIME':'t','CADENCENO':'cad'})
    ts = ts.drop('POS_CORR1 POS_CORR2'.split(),axis=1)
    ts['f'] = ts.f / ts.f.median() - 1

    # Normalize segment
    ts = np.array(pd.DataFrame(ts).to_records(index=False))

    if plot_diag:
        c = Circle(pos,radius=8,fc='none',ec='red')
        gca().add_artist(c)
        imshow2( np.median(cube['FLUX'],axis=0) )

    return ts

def r2fm(r,field):
    """
    Convienence Function. Make masked array from r['f'] and r['fmask']
    """
    return ma.masked_array(r[field],r['fmask'])
