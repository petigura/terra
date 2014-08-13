"""
Module for performing photometry of K2 data.


"""
import h5py
from numpy import ma
from matplotlib.pylab import *

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
    
def getkic(f,kic):
    """
    Read in K2 pixel data from h5 repo.
    """
    
    with h5py.File(f) as h5:
        b   = kic==h5['kic'][:].astype(int) 
        s   = f.split('/')[-1]
        if b.sum()==0:
            print '%s does not contain %i' % (s,kic)
        elif h5['ts'][b,0]['TIME'] < 1:
            print '%s does not contain %i' % (s,kic)
        else:
            print '%s %i' % (s,kic)
            cube = h5['cube'][b,:][0]
            ts = h5['ts'][b,:][0]
            cube = cube[c0_start_cad:]
            return ts,cube

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

def circular_photometry(kic,plot_diag=False):
    ts,cube = getkic('C0.h5',kic)

    cube = cube['FLUX']
    cube = flat_cube(cube)

    # pos = get_pos(cube,plot_diag=plot_diag)
    pos = (25,25)
    radius = 8.
    apertures = ('circular', radius)   

    nobs = cube.shape[0]
    mask = cube.sum(axis=1).sum(axis=1)==0
    f = ma.masked_array(np.zeros(nobs),mask)

    # Brightest sources

    for i in range(nobs):
        try:
            fluxtable, aux_dict = aperture_photometry(cube[i], pos, apertures)
            f.data[i] = fluxtable['aperture_sum']
        except:
            f.mask[i] = True

            

    return f



