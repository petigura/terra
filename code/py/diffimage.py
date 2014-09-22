"""
Module for computing difference images in and out of transit
"""
import cPickle as pickle
from argparse import ArgumentParser

import h5py
from scipy import ndimage as nd
import pandas as pd
from astropy.io import fits
from matplotlib.pylab import *

from image_registration import register_images
from pixel_decorrelation import loadPixelFile,get_stars_pix,plot_label,subpix_reg_stack
from photometry import imshow2

class baseObject:
    """Empty object container.
    """
    # 2010-01-24 15:13 IJC: Added to spitzer.py (from my ir.py)
    def __init__(self):
        return
    def __class__(self):
        return 'baseObject'

# The keys to pass around as object attribtutes
keys = 'dy dx dxs dys dfimi dfimo dfim mstack stack pixelfile gridfile'.split()

def make_diffimage(pixelfile,gridfile):
    # Load up transit information from grid file
    print "Reading transit parameters from %s" % gridfile
    h5 = h5py.File(gridfile)
    cal = h5['pp/cal'][:]
    lc = pd.DataFrame(dict(tPF=h5['tPF'],cad=cal['cad'],f=cal['f']))
    tdur = h5.attrs['tdur']

    print "Calculating relative offsets between frames"
    cube,headers = loadPixelFile(pixelfile)

    # Put images in to DataFrame

    flux = ma.masked_invalid(cube['flux'])
    flux = ma.masked_invalid(flux)
    flux.fill_value=0
    flux = flux.filled()

    medflux = ma.median(flux.reshape(flux.shape[0],-1),axis=1)
    flux = flux - medflux[:,newaxis,newaxis]

    flux = [f for f in flux]
    dfim = pd.DataFrame(dict(cad=cube['CADENCENO'],flux=flux))
    dfim['cad'] = dfim.cad.astype(int)

    # Only select images used in LC
    dfim = pd.merge(dfim,lc,on='cad')
    dfim.index=dfim.cad

    flux  = np.array(dfim.flux.tolist())
    dx,dy = subpix_reg_stack(flux) 

    dfim['dx'] = dx
    dfim['dy'] = dy

    fluxsL = []
    for i in range(len(dfim)):
        fluxs = nd.shift(flux[i],[-dy[i],-dx[i]],order=4)
        fluxsL +=[fluxs]
    fluxs = np.array(fluxsL) # Stack of the shifted images
    # dxs and dys are the offsets after the arrays have been shifted.
    dxs,dys = subpix_reg_stack(fluxs) 
    dfim['fluxs'] = [f for f in fluxs]

    dfimo = dfim[abs(dfim.tPF) < 0.5*tdur]
    dfimi = dfim[abs(dfim.tPF).between(0.5*tdur,1.0*tdur)]

    # Split up images into
    # i - in transit
    # o - out of transit
    # is - in transit, shifted
    # os - out of transit, shifted

    stack = {}
    stack['i'] = np.array(dfimi['flux'].tolist())
    stack['o'] = np.array(dfimo['flux'].tolist())
    stack['is'] = np.array(dfimi['fluxs'].tolist())
    stack['os'] = np.array(dfimo['fluxs'].tolist())

    mstack = {}
    stackkeys = 'i o is os'.split()
    for k in stackkeys:
        mstack[k] = median(stack[k],axis=0)

    # Shove working arrays into a big container object 
    o = baseObject()
    for k in keys:
        eval( 'setattr(o,"%s",%s)' % (k,k) )

    return o

def plot_diffimage(o):
    # Pull working arrays into current namespace
    for k in keys:
        exec( '%s = getattr(o,"%s")' % (k,k) )

    fig,axL = subplots(ncols=2,nrows=2,figsize=(8,8))
    sca(axL[0,0])
    plot(dx,dy,'.',label='Raw Images')
    plot(dxs,dys,'.',label='Images shifted by 4-order polynomial')
    xlabel('dx (pixels)')
    ylabel('dy (pixels)')
    legend()

    sca(axL[0,1])
    plot(dfimi.tPF,dfimi.f,'.',label='In Transit')
    plot(dfimo.tPF,dfimo.f,'.',label='Out of Transit')
    xlabel('t - t0 (days)')
    ylabel('flux')
    legend()

    sca(axL[1,0])
    refimage = dfim.iloc[0]['flux']
    catcut = get_stars_pix(pixelfile,refimage)
    epic = fits.open(pixelfile)[0].header['KEPLERID']
    logrefimage = log10(refimage)
    logrefimage = ma.masked_invalid(logrefimage)
    logrefimage.mask = logrefimage.mask | (logrefimage < 0)
    logrefimage.fill_value = 0 
    plot_label(logrefimage.filled(),catcut,epic,colorbar=False)
    title('log10( Reference Frame )')

    sca(axL[1,1])
    diffms = mstack['os'] - mstack['is']
    imshow2(diffms)
    plot_label(diffms,catcut,epic,colorbar=False)
    title('Difference Image\nOut of Transit - In Transit')
    gcf().set_tight_layout(True)

if __name__ == "__main__":
    parser = ArgumentParser(description='Ensemble calibration')
    parser.add_argument('pixelfile',type=str,help='*.fits file')
    parser.add_argument('gridfile',type=str,help='*.grid.h5 file')
    args  = parser.parse_args()
    
    o = make_diffimage(args.pixelfile,args.gridfile)
    plot_diffimage(o)
    plotfile = args.gridfile.replace('grid.h5','diffim.png')
    gcf().savefig(plotfile)
