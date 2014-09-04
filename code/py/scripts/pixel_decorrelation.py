"""
Routines & functions for analysis of K2 ("Kepler2") data.

To implement the "Vanderburg algorithm" of Vanderburg & Johnson (2014;
http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1408.3853), see the
main driver function :func:`runPixelDecorrelation` (for fixed aperture
size) the super-driver function :func:`runOptimizedPixelDecorrelation`
(to optimize the aperture size), or see the command-line example
below.


:EXAMPLES:
    To avoid hassles, now runs directly from the command line. To see a
    list of all command-line options, just enter:
     ::

      python k2.py --help

    Assuming the desired data file is in your current working directory, a
     3-core analysis of a K2 Engineering data field could be run via:
     ::

      python k2.py -f kplr060017806-2014044044430_lpd-targ.fits -x 25 -y 25 -n 3 -r 4 --tmin=1862.45 --minrad=3 --maxrad=10 --verbose=1

:REQUIREMENTS:
  Public Python modules:
    matplotlib / pylab
    NumPy
    SciPy
    AstroPy (or PyFITS)

  Private Python modules:
    analysis.py -- http://www.lpl.arizona.edu/~ianc/python/analysis.html
    tools.py -- http://www.lpl.arizona.edu/~ianc/python/tools.html
    phot.py -- http://www.lpl.arizona.edu/~ianc/python/phot.html


:HISTORY:
  2014-08-27 16:17 IJMC: Created.

  2014-09-02 18:48 IJMC: Spruced up and sent to E.P.
"""

# Import standard Python modules:
import matplotlib
matplotlib.use('Agg')  # Should allow plotting directly to files.

import numpy as np
import pylab as py
from scipy import interpolate
from multiprocessing import Pool
from optparse import OptionParser
import os
import sys
import warnings
warnings.simplefilter('ignore', np.RankWarning) # For np.polyfit
try:  
    from astropy.io import fits as pyfits
except:
    import pyfits

# Import my personal modules:
import analysis as an
import tools
import phot
try:
    import _chi2  # Leverage my c-based chi-squared routine:
    c_chisq = True
except:
    c_chisq = False


class baseObject:
    """Empty object container.
    """
    # 2010-01-24 15:13 IJC: Added to spitzer.py (from my ir.py)
    def __init__(self):
        return

    def __class__(self):
        return 'baseObject'

def main(argv=None):
    if argv is None:
        argv = sys.argv

    narg = len(argv)
    #_save = os.path.expanduser('~') + '/proj/transit/kepler/proc/'
    #_data = os.path.expanduser('~') + '/proj/transit/kepler/data/'

    _save = os.getcwd() + '/'
    _data = os.getcwd() + '/'

    if narg==1:  # E.g., running from within Python interpreter:
        # Use these 3 parameters for 'original' Kepler data:
        fn = _data + 'kplr009390653-2010265121752_lpd-targ.fits'
        tlimits = None
        xcen, ycen = 1,3

        # Use these 3 parameters for "K2" Kepler data:
        fn = _data + 'kplr060018007-2014044044430_lpd-targ.fits'
        tlimits = [1862.45, np.inf]
        xcen, ycen = 25, 25

        nthreads = 1
        resamp = 1  # photometry oversampling factor
        nordGeneralTrend = 5
        fs = 15
        minrad, maxrad = 1.5, 15
        gausscen = 0

        output = 1
        plotmode = 'texexec' 

    else:  # Probably running from the command line.
        p = OptionParser()

        p.add_option('-f', '--file', dest='fn', type='string', metavar='FILE',
                     help='Pixel data filename', action='store')
        p.add_option('-x', '--xcen', dest='xcen', type='float', 
                     help='Row (X-coordinate) of target in frame')
        p.add_option('-y', '--ycen', dest='ycen', type='float', 
                     help='Column (Y-coordinate) of target in frame')
        p.add_option('-n', '--nthreads', dest='nthreads', type='int', 
                     help='Number of threads for multiprocessing.', default=1)
        p.add_option('-r', '--resamp', dest='resamp', type='int', 
                     help='Resampling factor for partial-pixel photometry.', 
                     default=1)
        p.add_option('-g', '--gentrend', dest='nordGeneralTrend', type='int', 
                     help='Polynomial order for fitting general trend.', 
                     default=5)
        p.add_option('--minrad', dest='minrad', type='float', 
                     help='Minimum aperture radius (pixels)', default=1.5)
        p.add_option('--maxrad', dest='maxrad', type='float', 
                     help='Maximum aperture radius (pixels)', default=15)
        p.add_option('--tmin', dest='tmin', type='float', 
                     help='Minimum valid time index. (e.g., 1862.45 for K2 engineering)', default=1862.45)
        p.add_option('--tmax', dest='tmax', type='float', 
                     help='Maximum valid time index.', default=np.inf)
        p.add_option('--gausscen', dest='gausscen', type='int', 
                     help='Fit 2D gaussians for centroids if 1 (DEFAULT) or not (if 0)', default=1)
        p.add_option('--fs', '--fontsize', dest='fs', type='float', 
                     help='Font size for plots.', default=15)
        p.add_option('-d', '--datadir', dest='_data', type='string', 
                     metavar='DIR', help='Load data from this directory.', 
                     action='store', default=_data)
        p.add_option('-s', '--savedir', dest='_save', type='string', 
                     metavar='DIR', help='Save data into this directory.', 
                     action='store', default=_save)
        p.add_option('--plotalot', dest='plotalot', type='int', 
                     help='Set verbose generation of plots.', action='store', 
                     default=False)
        p.add_option('--verbose', dest='verbose', type='int', 
                     help='Set verbosity level for output text.', 
                     action='store', default=0)
        p.add_option('--output', dest='output', type='int', 
                     help='0=pickled dict (DEFAULT), 1=pickled object (less compatible), 2=FITS (limited data)', 
                     action='store', default=0)
        p.add_option('--plotmode', dest='plotmode', type='string', 
                     help="'texexec' or 'gs' are best, but 'tar' is safest.", 
                     action='store', default='tar')

        options, args = p.parse_args()

        fn       = options.fn
        xcen     = options.xcen
        ycen     = options.ycen
        nthreads = options.nthreads
        resamp   = options.resamp
        nordGeneralTrend = options.nordGeneralTrend
        minrad   = options.minrad
        maxrad   = options.maxrad
        tmin     = options.tmin
        tmax     = options.tmax
        fs       = options.fs
        _save    = options._save
        _data    = options._data
        plotalot = options.plotalot
        verbose  = options.verbose 
        gausscen = options.gausscen
        output   = options.output
        plotmode = options.plotmode

        tlimits = [tmin, tmax]

        if not os.path.isfile(fn):
            fn = _data + fn

    # Setup Pool for multiprocessing support:
    if nthreads==1:
        pool = None
    else:
        pool = Pool(processes=nthreads)

    # Run the code:
    results = runOptimizedPixelDecorrelation(fn, (xcen, ycen), resamp=resamp, nordGeneralTrend=nordGeneralTrend, pool=pool, tlimits=tlimits, verbose=verbose, plotalot=0, minrad=minrad, maxrad=maxrad, gausscen=gausscen)

    # Set up results filename:
    dapstr = (('%1.2f-'*3) % tuple(results.apertures)).replace('.', ',')[0:-1]
    locstr = 'loc-%i-%i' % tuple(np.array(results.loc).round())
    savefile = '%s%i_%s_r%s_m%i_p%i_s%i' % (_save, results.headers[0]['KEPLERID'], locstr, dapstr, results.nordArc, results.nordGeneralTrend, results.nordPixel1d)

    # Save everything to disk:
    if output==0 or output==1: # Save a pickle:
        picklefn = savefile + '.pickle'
        iter = 1
        while os.path.isfile(picklefn):
            picklefn = savefile + '_%i.pickle' % iter
            iter += 1
        if output==0: # Save a simple dict:
            tools.savepickle(tools.obj2dict(results), picklefn)
        elif output==1: # Save the object itself:
            tools.savepickle(results, picklefn)

    elif output==2: #Save a FITS file
        hdu1 = results2FITS(results)
        fitsfn = savefile + '.fits'
        iter = 1
        while os.path.isfile(fitsfn):
            fitsfn = savefile + '_%i.fits' % iter
            iter += 1
        hdu1.writeto(fitsfn)

        
    # Plot pretty pictures & print to disk:
    plotPixelDecorResults(results, fs=fs)

    pdffn = savefile + '.pdf'
    iter = 1
    while os.path.isfile(pdffn):
        pdffn = savefile + '_%i.pdf' % iter
        iter += 1

    tools.printfigs(pdffn, pdfmode=plotmode, verbose=verbose)
    py.close('all')
    return

import pandas as pd

def results2FITS(o):
    """Convert results from :func:`runOptimizedPixelDecorrelation` or
    :func:`runPixelDecorrelation` into a FITS Header unit suitable for
    writing to disk."""
    # 2014-09-02 18:44 IJMC: Created
    
    names = 'time cad rawFlux cleanFlux decorMotion decorBaseline bg x y arcLength noThrusterFiring'.split() 
    
    data = [getattr(o,n) for n in names]
    data = np.rec.fromarrays(data,names=names)
    hdu = fits.BinTableHDU(data=data)

    hdu.header['time'] = 'Time, BJD_TDB'
    hdu.header['rawFlux'] = 'Raw aperture photometry'
    hdu.header['cleanFlux'] = 'Cleaned, detrended photometry'
    hdu.header['decorMotion'] = 'Motion component of decorrelation.'
    hdu.header['decorBaseline'] = 'Long-term trend component of decorrelation.'
    hdu.header['bg'] = 'Background from photometry.'
    hdu.header['x'] = 'X motion (pixels).'
    hdu.header['y'] = 'Y motion (pixels).'
    hdu.header['arcLength'] = 'Length along arc (pixels).'
    hdu.header['noThrusterFiring'] = 'No Thruster Firing Identified'
    hdu.header['APRAD0'] = o.apertures[0]
    hdu.header['APRAD1'] = o.apertures[1]
    hdu.header['APRAD2'] = o.apertures[2]
    hdu.header['LOC1'] = o.loc[0]
    hdu.header['LOC2'] = o.loc[1]
    for key in ('kepmag', 'kid', 'resamp', 'rmsCleaned', 'rmsHonest', 'nordArc', 'nordPixel1d', 'nordGeneralTrend', 'rmsCleaned', 'rmsHonest'):
        val = getattr(o, key)
        if hasattr(val, '__iter__'):
            hdu.header[key] = val[0]
        else:
            hdu.header[key] = val

    return hdu

def runOptimizedPixelDecorrelation(fn, loc, apertures=None, resamp=1, nordGeneralTrend=5, verbose=False, plotalot=False, xy=None, tlimits=[1862.45, np.inf], nthreads=1, pool=None, minrad=2, maxrad=15, minSkyRadius=4, skyBuffer=2, skyWidth=3, niter=12, gausscen=True):
    """Run (1D) pixel-decorrelation of Kepler Data, and optimized
    aperture too. If you want a single, fixed aperture then use
    :func:`runPixelDecorrelation` instead.

    :INPUTS:
      The same as for :func:`runPixelDecorrelation`, with a few additions:

      minrad : positive scalar
        The smallest valid aperture radius, in pixels.

      maxrad : positive scalar
        The largest valid aperture radius, in pixels. For very faint
        targets, this should probably be set smaller than ~8.

      minSkyRadius : positive scalar
        Minimum radius of the inner sky annulus aperture, in pixels.

      minSkyBuffer : positive scalar
        Minimum buffer, in pixels, between the target aperture and the
        inner sky annulus aperture.

      skyBuffer : positive scalar
        Radius difference between inner sky annulus aperture and
        target aperture.

      skyWidth : positive scalar
        The width of the sky annulus aperture.

      gausscen : bool
        If True, fit 2D Gaussians to the PSFs as an additional
        centroiding mechanism.  This is unfortunately rather slow!

    :NOTES:
      This routine tried to optimize the photometric extraction
      aperture based on the RMS in the final light curve. Thus, it is
      poorly-suited to targets with rapid, high intrinsic variability.
    """
    # 2014-08-29 11:59 IJMC: Created

    
    def getAperRadii(targRad):
        if np.array(targRad).ndim>0:  targRad = targRad[0]
        targRad = max(min(targRad, maxrad), minrad)
        skyInner = max(targRad + skyBuffer, minSkyRadius)
        skyOuter = skyInner + skyWidth
        return tuple(np.array([targRad, skyInner, skyOuter]).squeeze())

    def nearlyIn(val1, seq, tol=0.1):
        seq = np.array(seq, copy=False)
        return (np.abs(seq - val1)<=tol).any()
            
    if verbose: 
        print "Starting K2 aperture-photometry optimization run."
        print " Filename is: " + fn

    outputs = []
    inner_ap_radii = np.arange(minrad, min(6, maxrad))
    if maxrad>6:
        nlog = np.int(np.log(maxrad/6.) / np.log(1.2))
        inner_ap_radii = np.concatenate((inner_ap_radii, 6*1.2**np.arange(nlog+1)))
        inner_ap_radii = inner_ap_radii[inner_ap_radii <= maxrad]

    for rap0 in inner_ap_radii:
        rap = getAperRadii(rap0)
        outputs.append(runPixelDecorrelation(fn, loc, rap, resamp=resamp, nordGeneralTrend=nordGeneralTrend, verbose=verbose, plotalot=plotalot, xy=xy, tlimits=tlimits, gausscen=gausscen, nthreads=nthreads, pool=pool))

    if verbose:
        print "Finished with initial grid of aperture sizes. Now home in."


    # Do a really crude homing-in algorithm. It's tough since the
    # underlying function is unlikely to be smooth, but there's still
    # room for improvement here:
    manyRMS = np.array([o.rmsHonest for o in outputs]).squeeze()
    minstep = 0.1
    thisstep = 1.
    for jj in range(niter):
        best1 = (np.array(manyRMS)==manyRMS.min()).nonzero()[0][0]
        best2 = np.array(manyRMS)<=np.sort(manyRMS)[1]
        best3 = np.array(manyRMS)<=np.sort(manyRMS)[2]
        RMSfit = np.polyfit(np.array(inner_ap_radii)[best3], manyRMS[best3], 2)
        nextguess = -0.5*RMSfit[1]/RMSfit[0]
        if RMSfit[0]<0 or nextguess<minrad or nextguess>maxrad or nextguess in inner_ap_radii:
            linfit = np.polyfit(np.array(inner_ap_radii)[best2], manyRMS[best2], 1)
            nextguess = inner_ap_radii[best1] - np.sign(linfit[0]) * max(minstep, np.abs(np.diff(np.array(inner_ap_radii)[best2][0:2])))
        

        rap = getAperRadii(nextguess)

        #print rap[0], np.sort(inner_ap_radii)
        thisIter = 0
        while nearlyIn(rap[0], inner_ap_radii, tol=0.1) and thisIter<100:
            nextguess = min(max(inner_ap_radii[best1] + thisstep*np.random.randn(), minrad), maxrad)
            thisstep = max(minstep, thisstep*0.5)
            rap = getAperRadii(nextguess)
            thisIter +=1
            if verbose>1:   print "Guess %1.2f basically already tried. Retrying." % rap[0]
            if thisIter==100:  thisStep = 0.5


        outputs.append(runPixelDecorrelation(fn, loc, rap, resamp=resamp, nordGeneralTrend=nordGeneralTrend, verbose=verbose, plotalot=plotalot, xy=xy, tlimits=tlimits, gausscen=gausscen, nthreads=nthreads, pool=pool))
        inner_ap_radii = np.concatenate((inner_ap_radii, [rap[0]]))
        manyRMS = np.array([o.rmsHonest for o in outputs]).squeeze()

        if verbose:
            print "Finished with iteration %i/%i. Last guess ap. %1.2f" % (jj+1, niter, rap[0])
        
            
    manyRMS = np.array([o.rmsHonest for o in outputs]).squeeze()
    cleanRMS = np.array([o.rmsCleaned for o in outputs]).squeeze()

    finalIndex = (manyRMS==manyRMS.min()).nonzero()[0][0]
    finalOutput = outputs[finalIndex]
    finalOutput.search_inner_ap_radii = inner_ap_radii
    finalOutput.search_rms = manyRMS
    finalOutput.search_rms_aggressive = cleanRMS

    return finalOutput


def runPixelDecorrelation(fn, loc, apertures, resamp=1, nordGeneralTrend=5, verbose=False, plotalot=False, xy=None, tlimits=[1862.45, np.inf], nthreads=1, pool=None, gausscen=True):
    """Run (1D) pixel-decorrelation of Kepler Data with a single aperture setting.

    :INPUTS:
      fn : str
        Filename of pixel file to load.

      loc: 2-sequence
        indexing coordinates of target. Need not be integers, but if
        converted to int these would index the approximate location of
        the target star.

        If loc is None, just use the brightest pixel. This is dangerous!

      apertures : various
        If a 3-sequence, this indicates the *radii* of three circular
        apertures (in pixels).  Passed to
        :func:`extractPhotometryFromPixelData`

      resamp : int
        Factor by which to interpolate frame before measuring photometry
        (in essence, does partial-pixel aperture photometry)

      nordGeneralTrend : int
        Polynomial order to remove slow, overall trends from the photometry.

      xy : 2-sequence of 1D NumPy arrays
        If you think you *know* the relative motions of your stars on
        the detector, you can save considerable time by inputting them
        here. GIGO.

      tlimits : 2-sequence of scalars
        Valid range for timestamps.  To exclude K2 early-engineering
        data, set to [1862.45, np.inf].  Set to None for no limits.

      gausscen : bool
        If True, fit 2D Gaussians to the PSFs as an additional
        centroiding mechanism.  This is unfortunately rather slow!

      nthreads : positive integer
        Set >1 for multithreaded processing.  At present this is only
        used for fitting 2D Gaussians in
        :func:`getCentroidsGaussianFit` (which takes up the most time).

      pool : multiprocessing.Pool() object
        If you like, pass in a ready-made Pool() object. I find this
        gives me fewer problems than using 'nthreads'.
        


    :OUTPUTS:
      A container-type object with many relevant fields. A few of the
      most important are demonstrated in the example below.

    :TO-DO:
      Add hooks for PRF fitting (to be implemented within
      :func:`extractPhotometryFromPixelData`)

      Maybe add option for outputs in other formats (FITS, array,
      dict, etc.)?

    :EXAMPLE:
     ::
      
      import k2
      import pylab as py
      import os

      _data = os.path.expanduser('~/proj/transit/kepler/data/')
      fn = _data + 'kplr060021426-2014044044430_lpd-targ.fits'
      xcen, ycen = 25, 25
      ap_radii = [5, 7, 9]
      resamp = 1

      # Run the analysis:
      output = k2.runPixelDecorrelation(fn, (xcen, ycen), ap_radii, resamp=resamp, nordGeneralTrend=5):

      # Plot the results:
      k2.plotPixelDecorResults(finalOutput, fs=15)
    """
    # 2014-08-28 10:20 IJMC: Created

    if verbose: 
        print "Starting K2 aperture-photometry run."
        print " Filename is:   " + fn
        print " Apertures are: " + str(apertures)
        print " verbosity level is: %i" % verbose

    # Load data:


    cube,headers = loadPixelFile(fn, header=True, tlimits=tlimits)
    time, data, edata = cube['time'],cube['flux'],cube['flux_err']

    # Extract photometry (add eventual hook for PRF fitting):
    flux, eflux, bg, testphot = extractPhotometryFromPixelData(data, loc, apertures, resamp=resamp, verbose=verbose, retall=True)

    if xy is None:
        # Compute centroid motions in 3 different ways:
        x0, y0 = xcorrStack(data, npix_corr=3)
        x0 = testphot.position[0] - x0
        y0 = testphot.position[1] - y0
        x1, y1 = getCentroidsStandard(data, mask=testphot.mask_targ, bg=bg)
        xys = [(x0,y0), (x1,y1)]
        if gausscen:
            x2, y2 = getCentroidsGaussianFit(data, testphot.position, testphot.mask_targ, flux=np.median(flux), bg=np.median(bg), errstack=edata, plotalot=plotalot, nthreads=nthreads, pool=pool, verbose=verbose)
            xys = xys + [(x2,y2)]
    else:
        xys = [(xy[0], xy[1])]

    # Detrend the flux, picking optimal values for all parameters:
    out = detrendFluxArcMotion(time, flux, xys, nordGeneralTrend=nordGeneralTrend, these_nord_arcs=None, these_nord_pixel1d=None, goodvals=None, plotalot=plotalot, verbose=verbose)
    
    # Stick everything into a big container-object:
    output = baseObject()
    output.headers = headers
    output.kid = headers[0]['KEPLERID']
    output.kepmag = headers[0]['KEPMAG']
    output.apertures = apertures
    output.nordGeneralTrend = nordGeneralTrend
  
    output.nordArc = out.nord_arc
    output.nordPixel1d = out.nord_pixel1d
    output.arc_fit = out.arc_fit
    output.decorBaseline = out.baseline
    output.decorMotion = out.decor
    if False: output.detrendSpline = out.detrendSpline
    output.x = out.x
    output.y = out.y
    output.rmsCleaned = out.rmsCleaned
    output.rmsHonest = out.rmsHonest
    output.arcLength = out.s
    output.noThrusterFiring = out.noThrusterFiring.squeeze()
    output.goodvals = out.goodvals

    output.time = time
    output.rawFlux = flux
    output.resamp = resamp
    output.cleanFlux = flux / out.decor / out.baseline
    output.bg = bg
    output.loc = testphot.position
    output.cad = cube['CADENCENO']

    output.medianFrame = np.median(data, axis=0)
    output.crudeApertureMask = testphot.mask_targ

    return output


def loadPixelFile(fn, tlimits=None, bjd0=2454833, header=False):
    """
    Convert a Kepler Pixel-data file into time, flux, and error on flux.

    :INPUTS:
      fn : str
        Filename of pixel file to load.

      tlimits : 2-sequence
        Valid time interval (before application of 'bjd0'). 

      bjd0 : scalar
        Value that will be added to the time index. The default
        "Kepler Epoch" is 2454833.

      header : bool
        Whether to also return the FITS headers.

    :OUTPUTS:
      time, datastack, data_uncertainties [, FITSheaders]
    """
    # 2014-08-27 16:23 IJMC: Created

    if tlimits is None:
        mintime, maxtime = -np.inf, np.inf
    else:
        mintime, maxtime = tlimits
        if mintime is None:
            mintime = -np.inf
        if maxtime is None:
            maxtime = np.inf

    f = pyfits.open(fn)


    cube = f[1].data
    flux0 = cube['flux'].sum(2).sum(1)

    index = (cube['quality']==0) & np.isfinite(flux0) \
            & (cube['time'] > mintime) & (cube['time'] < maxtime) 

    cube = cube[index]
    cube['time'][:] = cube['time'] + bjd0

    if header:
        ret = (cube,) + ([tools.headerToDict(el.header) for el in f],)

    f.close()
    return ret

def extractPhotometryFromPixelData(stack, loc, apertures, resamp=1, recentroid=True, verbose=False, retall=True):
    """
    :INPUTS:
      stack : 2D or 3D NumPy array.
        Stack of pixel data, e.g. from :func:`loadPixelFile`
      
      loc: 2-sequence
        indexing coordinates of target. Need not be integers, but if
        converted to int these would index the approximate location of
        the target star.
  
        If loc is None, just use the brightest pixel. This is dangerous!
  
      apertures : various
        If a 3-sequence, this indicates the *radii* of three circular apertures.
  
        Otherwise, we'll have to think of something fancier to do.
  
    :NOTES:
      May have problems for data whose motions are comparable to or
      larger than the PSF size.
      """
    # 2014-08-27 16:25 IJMC: Created
  
    if stack.ndim==2:
        stack = stack.reshape((1,)+stack.shape)

    nobs = stack.shape[0]
    frame0 = np.median(stack, axis=0)
    if loc is None:
        loc = (frame0==frame0.max()).nonzero()

    apertures = np.array(apertures)
    if len(apertures)==3:
        dap = np.array(apertures)*2
        mask = None
    elif apertures.shape==stack.shape or apertures.shape==frame0.shape:
        dap = None
        mask = apertures

    phot0 = phot.aperphot(frame0, pos=loc, dap=dap, mask=mask, resamp=1, retfull=True)

    if recentroid:
        if verbose: print "Refining centroid position:"
        maxlim = max(stack.shape[1:])
        x1d, y1d = np.arange(stack.shape[2]), np.arange(stack.shape[1])
        xx, yy = np.meshgrid(x1d, y1d)
        newloc = np.array(loc, copy=True)
        oldloc = newloc - 99
        locHist = [(-1,-1)]
        while np.abs(oldloc - newloc).max()>0.1 and not (tuple(newloc) in locHist[0:-1]):
            oldloc = newloc.copy()
            xcen = np.array((phot0.mask_targ*(frame0-phot0.bg) * xx).sum()/(phot0.mask_targ*(frame0-phot0.bg)).sum())
            ycen = np.array((phot0.mask_targ*(frame0-phot0.bg) * yy).sum()/(phot0.mask_targ*(frame0-phot0.bg)).sum())
            xcen = min(max(xcen, 0), maxlim)
            ycen = min(max(ycen, 0), maxlim)
            newloc = np.array([ycen,xcen], copy=True)
            phot0 = phot.aperphot(frame0, pos=newloc, dap=dap, mask=mask, resamp=1, retfull=True)
            if verbose: print "  New centroid location is: " + str(newloc)
            loc = tuple(newloc)
            locHist.append(tuple(newloc))


    phot1 = phot.aperphot(frame0, pos=loc, dap=dap, mask=mask, resamp=1, retfull=True)
    phot4mask = phot.aperphot(frame0, pos=loc, dap=dap, mask=mask, resamp=resamp, retfull=True)

    if verbose>=2:
        py.figure()
        py.imshow(np.log10(np.abs(1.+phot1.mask_targ * (phot1.frame-phot1.bg))))
        py.colorbar()
        py.title('log10(frame)')

    if mask is None:
        mask = phot4mask.mask_targ + phot4mask.mask_sky*2.0

    phots = [phot.aperphot(frame, pos=(xcen, ycen), dap=dap, mask=mask, resamp=resamp, retfull=False) for frame in stack]

    flux = np.zeros(nobs, dtype=float)
    eflux = np.zeros(nobs, dtype=float)
    bg = np.zeros(nobs, dtype=float)
    for ii, o in enumerate(phots):
        flux[ii] = o.phot
        eflux[ii] = o.ephot
        bg[ii] = o.bg

    ret = flux, eflux, bg
    if retall:  ret = ret + (phot1,)
    return ret

def getCentroidsGaussianFit(stack, position, mask, flux=None, bg=None, widths=None, errstack=None, plotalot=False, nthreads=1, pool=None, verbose=False):
    """Measure centroids via 2D gaussian fitting.  Pretty slow!

    :INPUTS:
      stack : 2D or 3D NumPy array
        Stack of pixel data, e.g. from :func:`loadPixelFile`

      position : 2-sequence
        Coordinates at which the target of interest is located.

      mask : 2D or 3D NumPy array
        Boolean mask, False for all pixels that should be utterly ignored.

      flux : scalar or None
        Best guess for the total flux in the target.

      bg : scalar or None
        Best guess for the sky background at the target.

      widths : 2-sequence or None
        Best guess for the x- and y-widths of the Gaussian.

      errstack : 2D or 3D NumPy array, or None
        Uncertainties in the data found in 'stack'

      nthreads : positive integer
        Set >1 for multithreaded processing. See Notes, and 'pool'
        description below.

      pool : multiprocessing.Pool() object
        If you like, pass in a ready-made Pool() object. I find this
        gives me fewer problems than using 'nthreads'.

    :OUTPUTS:
      x, y

    :NOTES:
      When nthreads>1, one sometimes encounters strange errors like
      "malloc: *** error for object 0x10baf9000: pointer being freed
      already on death-row".  For me, this occurs on my MacBook Pro
      when I begin Python via "pylab" but not via "ipython" (v13.1,
      Python v2.7.6) or "python" (v2.7.6).  Any help in diagnosing
      this would be greatly appreciated!
    """
    # 2014-08-27 17:57 IJMC: Created

    userPool = False
    if pool is not None:
        nthreads = pool._processes
        userPool = True
    elif nthreads>1:
        #print "\nMultithreading does not work yet! Running with just one thread.\n"
        pool = Pool(processes=nthreads)


    if stack.ndim==2:
        stack = stack.reshape((1,)+stack.shape)
    if errstack is None:
        errstack = np.ones(stack.shape)
    elif errstack.ndim==2:
        errstack = errstack.reshape((1,)+errstack.shape)

    if mask.ndim==2:
        mask = np.tile(mask, (stack.shape[0], 1, 1))

    nobs = stack.shape[0]
    frame0 = np.median(stack, axis=0)
    eframe0 = np.median(errstack, axis=0)
    mask0 = mask.mean(0)

    if bg is None:
        bg = np.median(stack.ravel())
    if flux is None:
        flux = frame
    if widths is None:
        widths = 0.5, 0.5

    x1d, y1d = np.arange(stack.shape[2]), np.arange(stack.shape[1])
    yy, xx = np.meshgrid(x1d, y1d)

    guess = flux, widths[0], widths[1], position[0], position[1], np.pi, bg
    #gmodel = an.gaussian2d_ellip(guess, xx, yy)
    maxsize = max(frame0.shape)
    uniformprior = [(0, np.inf), (0, np.inf), (0, np.inf), (0, maxsize), (0, maxsize), None, None]
    fitkw = dict(uniformprior=uniformprior)
    fitargs = (an.gaussian2d_ellip, xx, yy, frame0, mask0/eframe0**2, fitkw)
    fit = an.fmin(errfunc, guess, args=fitargs, full_output=True, disp=verbose)
    if plotalot>=1:
        fmodel = an.gaussian2d_ellip(fit[0], xx, yy)
        py.figure()
        py.subplot(131)
        py.imshow(np.log10(np.abs(mask0*(frame0-bg)+1)))
        py.subplot(132)
        py.imshow(np.log10(np.abs(mask0*(fmodel-bg)+1)))
        py.subplot(133)
        py.imshow(np.log10(np.abs(mask0*(frame0-fmodel)+1)))



    if nthreads==1:
        if verbose: print "Begining single-threaded Gaussian fitting."
        gaussfits = np.zeros((nobs, fit[0].size), dtype=float)
        gausschis = np.zeros(nobs, dtype=float)
        for ii in xrange(nobs):
            fitargs = (an.gaussian2d_ellip, xx[mask[ii]], yy[mask[ii]], stack[ii][mask[ii]], 1./errstack[ii][mask[ii]]**2, fitkw)
            thisfit = an.fmin(errfunc, fit[0], args=fitargs, full_output=True, disp=False)
            gaussfits[ii] = thisfit[0]
            gausschis[ii] = thisfit[1]
            #mstack[ii] = an.gaussian2d_ellip(thisfit[0], xx, yy)


    else:
        if verbose: print "Begining multithreaded Gaussian fitting."
        test_kw = dict(full_output=True, disp=False)

        HUGEfitargs = [[errfunc, fit[0], (an.gaussian2d_ellip, xx[mask[jj]], yy[mask[jj]], stack[jj][mask[jj]], 1./errstack[jj][mask[jj]]**2, fitkw), test_kw] for jj in xrange(nobs)]
        test_fits = pool.map(an.fmin_helper2, HUGEfitargs)
        gaussfits = np.array([tt[0] for tt in test_fits])
        
        if not userPool:
            pool.close()
            pool.join()

    return gaussfits[:,3:5].T

def getCentroidsStandard(stack, mask=None, bg=None, strictlim=True):
    """Measure centroids via center-of-mass formulae.

    :INPUTS:
      stack : 2D or 3D NumPy array
        Stack of pixel data, e.g. from :func:`loadPixelFile`

      mask : 2D or 3D NumPy array
        Boolean mask, False for all pixels that should be utterly ignored.
        Useful for ignoring the effects of nearby, bright stars.

      bg : scalar or None
        Best guess for the sky background at the target in each
        frame. Useful for getting the centroiding just right.

      strictlim : bool
        If True, ensure that the resulting x,y values do not lie
        outside the dimensions of 'stack'

    :OUTPUTS:
      x, y
    """
    # 2014-08-27 18:17 IJMC: Created
    if stack.ndim==2:
        stack = stack.reshape((1,)+stack.shape)
    else:
        stack = stack.copy()

    nobs = stack.shape[0]
    if mask is None:
        mask = np.ones(stack.shape)
    elif mask.ndim==2:
        mask = np.tile(mask, (nobs, 1, 1))

    if bg is None:
        bg = np.zeros(stack.shape[0])

    
    stack -= bg.reshape(nobs, 1, 1)

    x1d, y1d = np.arange(stack.shape[2]), np.arange(stack.shape[1])
    xx, yy = np.meshgrid(x1d, y1d)

    denoms = (mask * stack).sum(2).sum(1)
    y2 = (mask * stack * xx).sum(2).sum(1) / denoms
    x2 = (mask * stack * yy).sum(2).sum(1) / denoms

    if strictlim:
        maxlim = max(stack.shape[1:])
        x2 = np.vstack((x2, np.zeros(nobs))).max(0)
        x2 = np.vstack((x2, maxlim*np.ones(nobs))).min(0)
        y2 = np.vstack((y2, np.zeros(nobs))).max(0)
        y2 = np.vstack((y2, maxlim*np.ones(nobs))).min(0)

    return x2, y2


def getArcLengths(time, xypairs, these_nord_arcs=None, verbose=False):
    """Measure the length along curved arcs, defined by x-y coordinates.
  
    :INPUTS:
      time : 1D NumPy array
        Time index at which X, Y values were measured.
  
      xypairs: sequence of 2-tuples 
        Sets of X & Y values, measured by various techniques.
  
      these_nord_arcs : sequence of positive ints
        Number of polynomial terms to use for fitting the rotated y'
        positions as a function of x'. Thus, zero is invalid. Each
        specified order will be tried. If None, defaults to
        np.arange(1,15)
    """
    # 2014-08-27 19:23 IJMC: Created

    if verbose: print "Measuring arc lengths for each set of (x,y)"

    nxy = len(xypairs)
    nobs = xypairs[0][0].size
    
    if these_nord_arcs is None:
        these_nord_arcs = np.arange(1, 15)

    nord = these_nord_arcs.size

    ss = np.zeros((nxy, nobs), dtype=float)
    nord_arcs = np.zeros(nxy, dtype=int)
    arc_fits = []
    thrusterFiringIndex = np.zeros((nxy, nobs), dtype=bool)
    for ii,xy in enumerate(xypairs):
        x, y = xy
        ### Rotate coordinate frames:
        rot,sjunk,vjunk = np.linalg.svd(np.cov(x, y))
        if rot[0,0]<0: rot *= -1
        xp, yp = np.dot(rot, np.vstack((x, y)))
        if xp.std()==0:
            xp_norm = np.ones(nobs, dtype=float)
        else:
            xp_norm = 2 * ((xp - xp.min()) / (xp.max() - xp.min()) - 0.5)


        ### Fit arc motions, iteratively rejecting outliers:
        arc_RMSes = np.zeros(nord)
        arc_npts = np.zeros(nord, int)
        these_arc_fits = []

        for jj in these_nord_arcs:
            xvecs = np.array([xp_norm**n for n in xrange(jj)]).T
            weights = np.ones(nobs)
            lastNgood = nobs+1
            while lastNgood > weights.sum():
                lastNgood = weights.sum()
                fit, efit = an.lsq(xvecs, yp, w=weights, checkvals=False)
                ypfit = np.dot(fit, xvecs.T)
                weights = np.abs(yp-ypfit) < (an.dumbconf(yp-ypfit, .683)[0]*5)
            arc_RMSes[jj-1] = (yp-ypfit)[weights].std()
            arc_npts[jj-1] = weights.sum()
            these_arc_fits.append(fit)
            #plot(xp, ypfit, '.')

        minRMS = arc_RMSes[np.isfinite(arc_RMSes)].min()
        arc_BICs = these_nord_arcs * np.log(arc_npts) + arc_npts * (arc_RMSes/minRMS)**2
        minBIC = arc_BICs[np.isfinite(arc_BICs)].min()
        nord_arcs[ii] = these_nord_arcs[arc_BICs==minBIC]
        fit = np.array(these_arc_fits)[arc_BICs==minBIC][0]
        arc_fits.append(fit)


        # Measure distance along arc:
        ss[ii] =  computePolyCurveDistance(xp_norm, fit) * (xp.max()-xp.min())/2.

        # Compute derivatives:
        dsdt = np.concatenate(([0], np.diff(ss[ii]) / np.diff(time)))
        # Mask outliers in derivative space ("thruster firings"):
        noFiring = np.ones(nobs, dtype=bool)
        for kk in xrange(10):
            noFiring = (np.abs(dsdt-np.median(dsdt)) < (an.dumbconf(dsdt[noFiring], .683)[0]*5))
        thrusterFiringIndex[ii] = True-noFiring


    return ss, thrusterFiringIndex, nord_arcs, arc_fits

def computePolyCurveDistance(x, coef, oversamp=100):
    """Compute distance along a polynomial curve, defined by input
    coordinates x and coefficients 'coef'.

    Specifically, we return:
      Integral_xmin^xi sqrt(1 + (df/dx)^2) dx
    for each point xi in x.
    """
    # 2014-08-26 10:29 IJMC: Created

    nobs = x.size
    nord = len(coef)

    x1fine = np.linspace(x.min(), x.max(), nobs*oversamp)
    dx = np.diff(x1fine).mean()
    xvecs1fine = np.vstack([x1fine**n for n in range(nord)]).T
    y1pfine = np.dot(coef, xvecs1fine.T)
    s1fine = np.concatenate(([0], np.cumsum(np.sqrt(1 + (np.diff(y1pfine)/dx)**2) * dx)))
    return np.interp(x, x1fine, s1fine)

#def estimatePhotometricTrend(time, flux, nord=5, goodvals=None, plotalot=False):
#    """
#    Documentation TBD -- 
#
#    """
#    # 2014-08-27 19:50 IJMC: Created
#
#    nobs = time.size
#    if goodvals is None:
#        goodvals = np.ones(nobs, dtype=bool)
#    else:
#        goodvals = np.array(goodvals, copy=True)
#
#    outliers = np.ones(nobs, bool)
#    while outliers[goodvals].any():
#        #spl = interpolate.UnivariateSpline(time[goodvals], flux[goodvals], w=1./eflux[goodvals]**2, k=5, s=nobs/4.)
#        pfit = np.polyfit(time[goodvals]-time.min(), flux[goodvals], nord)
#        baseline = np.polyval(pfit, time-time.min())
#        residuals = flux - baseline
#        outliers = np.abs(residuals) > (an.dumbconf(residuals[goodvals], .683)[0]*3)
#        goodvals[outliers] = False
#        if plotalot:
#            py.figure()
#            py.subplot(211)
#            py.plot(time, flux, '.k', time[goodvals], flux[goodvals], 'or', time, polyval(pfit, time-time.min()), '.c')
#            py.subplot(212)
#            py.plot(time, np.abs(residuals), '.k', time[goodvals], np.abs(residuals)[goodvals], 'or')
#            py.plot(xlim(), [(an.dumbconf(residuals[goodvals], .683)[0]*3)]*2, '--')
#
#    fluxDetrend = flux / baseline
#
#    return fluxDetrend, goodvals


def detrendFluxArcMotion(time, flux, xys, nordGeneralTrend=None, these_nord_arcs=None, these_nord_pixel1d=None, goodvals=None, plotalot=False, verbose=False):
    """
    See function name: remove effect of (projected 1D) motion on photometry.

    :INPUTS:
      time : 1D NumPy array
        Time index at which flux, X, Y values were measured.

      flux : 1D NumPy array
        Raw flux of the target at each timestep, e.g. from aperture photometry.

      xys: sequence of 2-tuples 
        Sets of X & Y values, measured by various techniques.

      nordGeneralTrend : int
        Polynomial order to remove slow, overall trends from the photometry.

      these_nord_arcs : sequence of positive ints
        Number of polynomial terms to use for fitting the rotated y'
        positions as a function of x'. Thus, zero is invalid. Each
        specified order will be tried. Passed to :func:`getArcLengths.

      these_nord_pixel1d : sequence of positive ints
        Number of polynomial terms to use for fitting the dependence
        of 'flux' on the projected arc motions from x and y. If None,
        defaults to range(3, 20)

      goodvals : 1D Numpy array (of type bool)
        True for usable mesurements; False elsewhere.
    """
    # 2014-08-27 20:06 IJMC: Created


    # Compute the arc lengths:
    ss, thrusterIndex, nord_arcs, arc_fits = getArcLengths(time, xys, these_nord_arcs, verbose=verbose)

    # Parse inputs:
    nxy = len(xys)
    nobs = time.size
    if these_nord_pixel1d is None:
        these_nord_pixel1d = np.arange(3, 20)
    if not isinstance(these_nord_pixel1d, np.ndarray):
        these_nord_pixel1d = np.array(these_nord_pixel1d)
    n1d = these_nord_pixel1d.size
    if goodvals is None:
        goodvals0 = True - thrusterIndex
    elif goodvals.ndim==1:
        goodvals0 = np.tile(goodvals, (nxy, 1))
    else:
        goodvals0 = goodvals

    # Detrend flux:
    goodvals = goodvals0.all(axis=0)
    if False:
        fluxDetrend, notOutliers = estimatePhotometricTrend(time, flux, nord=nordGeneralTrend, goodvals=goodvals0.all(axis=0))

    # Fit detrended flux to (projected) stellar motion:
    baseline_decors = np.zeros((nxy, nobs), dtype=float)
    RMSes = np.zeros(nxy)
    detrendSplines = []
    allRMSes = np.zeros((nxy, n1d), dtype=float)
    honestRMSes = np.zeros((nxy, n1d), dtype=float)
    all_sff_decors = np.zeros((nxy, n1d, nobs), dtype=float)
    all_baseline_decors = np.zeros((nxy, n1d, nobs), dtype=float)

    if verbose: print "Fitting combined (baseline*motion) decorrelation."
    for ii,this_s in enumerate(ss):
        if verbose>1: print "Starting iteration %i/%i." % (ii+1, len(ss))
        #nord_pixel1ds = np.zeros(n1d-1, dtype=int)
        these_splines = []
        #these_decors = np.zeros((nord_pixel1ds.size, nobs), dtype=float)
        for jj, nord in enumerate(these_nord_pixel1d):
            if verbose>1: print "Starting sub-iteration %i/%i." % (jj+1, len(these_nord_pixel1d))
            if False:   # Like Vanderburg & Johnson:
                fbins = np.linspace(this_s[goodvals].min(), this_s[goodvals].max(), nord)
                sbin,fbin,junk,efbin = tools.errxy(this_s[goodvals], fluxDetrend[goodvals], fbins, clean=dict(nsigma=3, remove='both', niter=1), yerr='std')
                finind = np.isfinite(sbin * fbin)
                these_splines.append(interpolate.UnivariateSpline(sbin[finind], fbin[finind], s=0, k=1))
                all_sff_decors[ii,jj] = these_splines[jj](this_s)
                fluxFinal = flux / all_sff_decors[ii,jj]
                pfit_decor = an.polyfitr(time[goodvals]-time.min(), (flux/all_sff_decors[ii,jj])[goodvals], nordGeneralTrend, 3)
                all_baseline_decors[ii,jj] = np.polyval(pfit_decor, time-time.min())
            else:   # Seems more reliable:
                t_norm = 2 * ((time - time.min()) / (time.max() - time.min()) - 0.5)
                s_norm = 2 * ((this_s - this_s.min()) / (this_s.max() - this_s.min()) - 0.5)
                #svecs = np.array([this_s**n for n in range(nord)]).T
                #tvecs = np.array([t_norm**n for n in range(nordGeneralTrend)]).T

                newchi = 9e99
                dchi = 9e99
                #sz = svecs[goodvals].mean(0)
                seval = np.ones(nobs)
                iter = 0
                maxiter = 2000
                while np.abs(dchi)>0.1 and iter < maxiter:
                    #tfit, etfit = an.lsq(tvecs[goodvals], (flux / seval)[goodvals])
                    #teval = np.dot(tvecs, tfit)
                    #sfit, esfit = an.lsq(svecs[goodvals], (flux / teval)[goodvals])
                    #seval = np.dot(svecs, sfit)
                    tfit = np.polyfit(t_norm[goodvals], (flux / seval)[goodvals], nordGeneralTrend-1)
                    teval = np.polyval(tfit, t_norm)
                    if (teval==0).any():  tevel[teval==0] = 1.
                    sfit = np.polyfit(s_norm[goodvals], (flux / teval)[goodvals], nord-1)
                    seval = np.polyval(sfit, s_norm)
                    if (seval==0).any():  sevel[seval==0] = 1.
                    oldchi = newchi + 0.
                    newchi = ((flux - seval * teval)**2)[goodvals].sum()
                    dchi = oldchi - newchi
                    iter += 1
                    #print iter, newchi, oldchi

                if iter==maxiter and verbose>1:
                    print "Did not converge on a (baseline*motion) fit after %i iterations." % iter
                #allvecs = np.hstack((svecs, tvecs))
                #bigfit, ebigfit = an.lsq(allvecs, np.log(flux))

                #bigS = np.dot(svecs, bigfit[0:nord])
                #bigT = np.dot(tvecs, bigfit[nord:])
                all_baseline_decors[ii,jj] = teval
                all_sff_decors[ii,jj] = seval

            allRMSes[ii,jj] = (flux/all_sff_decors[ii,jj]/ all_baseline_decors[ii,jj])[goodvals].std()
            honestRMSes[ii,jj] = (flux/all_sff_decors[ii,jj]/ all_baseline_decors[ii,jj])[True - thrusterIndex[ii]].std()

            
            #abc = np.dot(allvecs, bigfit)

            if plotalot>=2:
                py.figure()
                py.plot(this_s[goodvals], fluxDetrend[goodvals], 'oc')
                py.plot(this_s, all_sff_decors[ii,jj], '.k', mfc='orange')
                py.errorbar(sbin, fbin, efbin, fmt='ok', ms=11)
                py.title('%i, %i' % (ii, jj))

        detrendSplines.append(these_splines)
        minRMS = allRMSes[np.isfinite(allRMSes)].min()
        allChisq = goodvals.sum() * (allRMSes/minRMS)**2
        allBIC = allChisq + np.log(goodvals.sum()) * these_nord_pixel1d

    # Select the model with the lowest final RMS:
    bestBIC = allBIC[np.isfinite(allBIC)].min()
    bestInd = (allBIC==bestBIC).nonzero()


    output = baseObject()
    output.bestInd = bestInd
    output.x, output.y = xys[bestInd[0]]
    output.s = ss[bestInd[0]].squeeze()
    output.rmsCleaned = allRMSes[bestInd]
    output.rmsHonest = honestRMSes[bestInd]
    output.decor = all_sff_decors[bestInd].squeeze()
    output.nord_arc = nord_arcs[bestInd[0]]
    output.nord_pixel1d = (these_nord_pixel1d[bestInd[1]]-1)
    output.arc_fit = arc_fits[bestInd[0]]
    output.baseline = all_baseline_decors[bestInd].squeeze()
    if False: output.detrendSpline = detrendSplines[bestInd[0]][bestInd[1]]
    output.flux = flux
    output.goodvals = goodvals
    output.noThrusterFiring = True - thrusterIndex[bestInd[0]]

    if plotalot>=1:
        py.figure()
        py.plot(output.s, flux / output.baseline, 'or', mec='k')
        py.plot(output.s[goodvals], (flux / output.baseline)[goodvals], 'oc', mec='k')
        #plot(sbin, fbin, 'ok', ms=14)
        py.plot(output.s, output.decor, '.', color='orange')
        py.xlabel('Arclength [pixels]', fontsize=15)
        py.ylabel('Detrended Flux', fontsize=15)
        py.minorticks_on()


    return output

def plotPixelDecorResults(input, fs=15):
    """Plot the results of a pixel-decorrelation run.
    
    :INPUTS:
      input : object
        The resulting output from a successful call to
        :func:`runPixelDecorrelation`

      fs : positive scalar
        The font size.

    """
    # 2014-08-28 20:37 IJMC: Created

    titstr = 'EPIC %i, Kp=%1.2f' % (input.headers[0]['KEPLERID'], input.headers[0]['KEPMAG'])

    if hasattr(input, 'search_inner_ap_radii') and \
            hasattr(input, 'search_rms'):

        ai = np.argsort(input.search_inner_ap_radii)
        py.figure()
        ax1 = py.subplot(111, position=[.15, .15, .8, .75])
        py.semilogy(np.array(input.search_inner_ap_radii)[ai], input.search_rms[ai]*1e6, 'oc-')
        py.xlabel('Target Aperture Radius [pixels]', fontsize=fs)
        py.ylabel('RMS [ppm]', fontsize=fs)
        py.minorticks_on()
        #loglim = np.log10(np.array(py.ylim()))
        #abc = (np.logspace(*loglim, num=4*np.diff(loglim)+1))
        minmax = input.search_rms.min(), input.search_rms.max()
        factor = minmax[0] / minmax[1]
        yval = 1e6*(minmax[0] * np.sqrt(factor)), 1e6*(minmax[1] / np.sqrt(factor))
        py.ylim(yval)
        yticks = np.logspace(np.log10(yval[0]), np.log10(yval[1]), 8)
        py.yticks(yticks, ['%i' % el for el in yticks])
        ax = py.axis()
        py.plot([input.apertures[0]]*2, py.ylim(), '--k')
        py.axis(ax)
        py.title(titstr)
        py.text(.95, .9, 'Minimum: %1.1f ppm at R=%1.2f pixels' % (input.rmsHonest*1e6, input.apertures[0]), transform=ax1.transAxes, horizontalalignment='right', fontsize=fs)


    time = input.time
    if (time > 2454833).any():
        time -= 2454833
    fluxDetrend = input.rawFlux / input.decorBaseline

    py.figure(tools.nextfig(), [10, 6.5])
    ax1 = py.subplot(311, position=[.48, .65, .43, .2])
    py.plot(time, fluxDetrend, '.-k', mfc='c')
    py.ylabel('Normalized Flux', fontsize=fs)
    ax1.get_yaxis().set_major_formatter(py.FormatStrFormatter('%1.4f'))
    ax2 = py.subplot(312, position=[.48, .4, .43, .2])
    py.plot(time, input.x, '.-k', mfc='r')
    py.ylabel('X motion [pix]', fontsize=fs)
    ax3 = py.subplot(313, position=[.48, .15, .43, .2])
    py.plot(time, input.y, '.-k', mfc='r')
    py.ylabel('Y motion [pix]', fontsize=fs)
    py.xlabel('BJD - 2454833', fontsize=fs)

    ax4 = py.subplot(121, position=[.05, .1, .35, .7], aspect='equal')
    py.imshow(np.log10(input.medianFrame), aspect='equal', cmap=py.cm.cubehelix)
    py.ylim(py.ylim()[::-1])
    tools.drawCircle(input.loc[1], input.loc[0], input.apertures[0], color='lime', fill=False, linewidth=3)
    py.title('log10(median image)', fontsize=fs)
    py.gcf().text(.5, .95, titstr, fontsize=fs*1.2, horizontalalignment='center')
    ax = py.axis()
    py.plot(input.y, input.x, '.r')
    py.axis(ax)
    py.colorbar(orientation='horizontal')
    for ax in [ax1, ax2,ax3]:
        ax.get_xaxis().set_major_formatter(py.FormatStrFormatter('%i'))
        ax.minorticks_on()
        ax.get_yaxis().set_label_position('right')
        ax.get_yaxis().set_ticks_position('right')
        ax.get_yaxis().set_ticks_position('both')


    rot,sjunk,vjunk = np.linalg.svd(np.cov(input.x, input.y))
    if rot[0,0]<0: rot *= -1
    xp, yp = np.dot(rot, np.vstack((input.x, input.y)))
    if xp.std()==0:
        xp_norm = np.ones(nobs, dtype=float)
    else:
        xp_norm = 2 * ((xp - xp.min()) / (xp.max() - xp.min()) - 0.5)
    xvecs = np.array([xp_norm**n for n in range(input.nordArc)]).T
    ypfit = np.dot(input.arc_fit, xvecs.T)

    py.figure(tools.nextfig(), [10, 5])
    ax1 = py.subplot(121, position=[.1, .15, .35, .75])
    py.plot(input.x, input.y, 'oc', mec='k')
    py.xlabel('X motion [pixels]', fontsize=fs)
    py.ylabel('Y motion [pixels]', fontsize=fs)
    py.minorticks_on()
    ax2 = py.subplot(122, position=[.6, .15, .35, .75])
    py.plot(xp - xp.mean(), yp-ypfit.mean(), 'oc', mec='k')
    py.plot(xp-xp.mean(), ypfit-ypfit.mean(), '.r')
    py.xlabel('Projected X motion [pix]', fontsize=fs)
    py.ylabel('Projected Y motion [pix]', fontsize=fs)
    py.minorticks_on()
    py.text(.95, .05, 'BIC-optimal detrending order: %i' % (input.nordArc-1), transform=ax2.transAxes, horizontalalignment='right', fontsize=fs)
    py.gcf().text(.5, .95, titstr, fontsize=fs*1.2, horizontalalignment='center')

    sss = np.linspace(input.arcLength.min(), input.arcLength.max(), 1000)
    py.figure()
    ax1 = py.subplot(111, position=[.15, .15, .8, .75])
    py.plot(input.arcLength, fluxDetrend, 'or', mec='k')
    py.plot(input.arcLength[input.goodvals], fluxDetrend[input.goodvals], 'oc', mec='k')
    py.plot(input.arcLength, input.decorMotion, '.', color='orange', linewidth=3)
    #errorbar(sbin, fbin, efbin, fmt='ok', ms=10)
    ax = py.axis()
    #[plot([fb]*2, ax[2:], '--k') for fb in fbins]
    py.axis(ax)
    py.xlabel('Arclength [pixels]', fontsize=fs)
    py.ylabel('Detrended Flux', fontsize=fs)
    py.title(titstr, fontsize=fs*1.2)
    py.minorticks_on()
    ax1.get_yaxis().set_major_formatter(py.FormatStrFormatter('%01.4f'))
    py.text(.95, .05, 'BIC-optimal detrending order: %i' % (input.nordPixel1d), transform=ax1.transAxes, horizontalalignment='right', fontsize=fs)


    f0 = np.median(input.rawFlux)
    py.figure()
    ax1 = py.subplot(211, position=[.15, .45, .8, .45])
    py.plot(time, input.decorBaseline * input.decorMotion/f0, '-k')
    py.plot(time, input.rawFlux/f0, 'oc', mec='k')
    py.plot(time[input.noThrusterFiring], (input.rawFlux / input.decorMotion)[input.noThrusterFiring]/f0 - 8*input.rmsCleaned, 'o', mfc='orange', mec='k')
    py.ylabel('Normalized Flux', fontsize=fs)

    py.title(titstr, fontsize=fs*1.2)
    ax2 = py.subplot(212, position=[.15, .1, .8, .3])
    py.plot(time[input.noThrusterFiring], input.cleanFlux[input.noThrusterFiring], 'o', mfc='lime')
    py.plot(py.xlim(), [1,1], '--k')
    #ylim(1.-8*rms, 1.+8*rms)
    ax2.get_yaxis().set_major_formatter(py.FormatStrFormatter('%01.4f'))
    py.xlabel('BJD - 2454833', fontsize=fs)
    py.ylabel('Cleaned Flux', fontsize=fs)
    for ax in [ax1, ax2]:
        ax.get_xaxis().set_major_formatter(py.FormatStrFormatter('%i'))
        ax.minorticks_on()


    # Plot statistics for fit:
    runningStd6 = an.stdfilt(input.cleanFlux[input.goodvals], wid=13)
    RMS6hr = np.median(runningStd6)/np.sqrt(13)
    runningStd2 = an.stdfilt(input.cleanFlux[input.goodvals], wid=5)
    RMS2hr = np.median(runningStd6)/np.sqrt(5)

    tlines = ['EPIC %i' % input.headers[0]['KEPLERID'], 
              'Kp = %1.2f mag' % input.headers[0]['KEPMAG'],
              'Aperture radii = (%1.2f, %1.2f, %1.2f) pix' % tuple(input.apertures),
              'Photometry oversamp = %1.1f' % input.resamp,
              'Poly. order, target motion fit:       %i' % (input.nordArc-1),
              'Poly. order, photometric poly:        %i' % input.nordGeneralTrend,
              'Poly. order, pixel-motion correction: %i' % (input.nordPixel1d-1),
              '', 'Per-point RMS (honest):   %1.1f ppm' % (input.rmsHonest*1e6),
              '', 'Per-point RMS (cleaned):  %1.1f ppm' % (input.rmsCleaned*1e6),
              '2-hour RMS (cleaned):     %1.1f ppm' % (RMS2hr*1e6),
              '6-hour RMS (cleaned):     %1.1f ppm' % (RMS6hr*1e6)]

    #          'Optimal centroiding method employed: %i' % bestInd[0],

    fig, ax = tools.textfig(tlines, fontsize=15)

    return


def errfunc(*arg, **kw):
    """ Generic function to give the chi-squared error on a generic
        function or functions:

    :INPUTS:
       (fitparams, function, arg1, arg2, ... , depvar, weights)

      OR:
       
       (fitparams, function, arg1, arg2, ... , depvar, weights, kw)

      OR:
       
       (allparams, (args1, args2, ..), npars=(npar1, npar2, ...))

       where allparams is an array concatenation of each functions
       input parameters.

      If the last argument is of type dict, it is assumed to be a set
      of keyword arguments: this will be added to errfunc2's direct
      keyword arguments, and will then be passed to the fitting
      function **kw.  This is necessary for use with various fitting
      and sampling routines (e.g., kapteyn.kmpfit and emcee.sampler)
      which do not allow keyword arguments to be explicitly passed.
      So, we cheat!  Note that any keyword arguments passed in this
      way will overwrite keywords of the same names passed in the
      standard, Pythonic, way.


    :OPTIONAL INPUTS:
      jointpars -- list of 2-tuples.  
                   For use with multi-function calling (w/npars
                   keyword).  Setting jointpars=[(0,10), (0,20)] will
                   always set params[10]=params[0] and
                   params[20]=params[0].

      gaussprior -- list of 2-tuples (or None values), same length as "fitparams."
                   The i^th tuple (x_i, s_i) imposes a Gaussian prior
                   on the i^th parameter p_i by adding ((p_i -
                   x_i)/s_i)^2 to the total chi-squared.  Here in
                   :func:`devfunc`, we _scale_ the error-weighted
                   deviates such that the resulting chi-squared will
                   increase by the desired amount.

      uniformprior -- list of 2-tuples (or 'None's), same length as "fitparams."
                   The i^th tuple (lo_i, hi_i) imposes a uniform prior
                   on the i^th parameter p_i by requiring that it lie
                   within the specified "high" and "low" limits.  We
                   do this (imprecisely) by multiplying the resulting
                   deviates by 1e9 for each parameter outside its
                   limits.

      ngaussprior -- list of 3-tuples of Numpy arrays.
                   Each tuple (j_ind, mu, cov) imposes a multinormal
                   Gaussian prior on the parameters indexed by
                   'j_ind', with mean values specified by 'mu' and
                   covariance matrix 'cov.' This is the N-dimensional
                   generalization of the 'gaussprior' option described
                   above. Here in :func:`devfunc`, we _scale_ the
                   error-weighted deviates such that the resulting
                   chi-squared will increase by the desired amount.

                   For example, if parameters 0 and 3 are to be
                   jointly constrained (w/unity means), set: 
                     jparams = np.array([0, 3])
                     mu = np.array([1, 1])
                     cov = np.array([[1, .9], [9., 1]])
                     ngaussprior=[[jparams, mu, cov]]  # Double brackets are key!

      scaleErrors -- bool
                   If True, fit for the measurement uncertainties.
                   In this case, the first element of 'fitparams'
                   ("s") is used to rescale the measurement
                   uncertainties. Thus weights --> weights/s^2, and
                   chi^2 --> 2 N log(s) + chi^2/s^2 (for N data points).  


    EXAMPLE: 
      ::

       from numpy import *
       import phasecurves
       def sinfunc(period, x): return sin(2*pi*x/period)
       snr = 10
       x = arange(30.)
       y = sinfunc(9.5, x) + randn(len(x))/snr
       guess = 8.
       period = optimize.fmin(phasecurves.errfunc,guess,args=(sinfunc,x, y, ones(x.shape)*snr**2))
    """
    # 2009-12-15 13:39 IJC: Created
    # 2010-11-23 16:25 IJMC: Added 'testfinite' flag keyword
    # 2011-06-06 10:52 IJMC: Added 'useindepvar' flag keyword
    # 2011-06-24 15:03 IJMC: Added multi-function (npars) and
    #                        jointpars support.
    # 2011-06-27 14:34 IJMC: Flag-catching for multifunc calling
    # 2012-03-23 18:32 IJMC: testfinite and useindepvar are now FALSE
    #                        by default.
    # 2012-05-01 01:04 IJMC: Adding surreptious keywords, and GAUSSIAN
    #                        PRIOR capability.
    # 2012-05-08 16:31 IJMC: Added NGAUSSIAN option.
    # 2012-10-16 09:07 IJMC: Added 'uniformprior' option.
    # 2013-02-26 11:19 IJMC: Reworked return & concatenation in 'npars' cases.
    # 2013-03-08 12:54 IJMC: Added check for chisq=0 in penalty-factor cases.
    # 2013-04-30 15:33 IJMC: Added C-based chi-squared calculator;
    #                        made this function separate from devfunc.
    # 2013-07-23 18:32 IJMC: Now 'ravel' arguments for C-based function.
    # 2013-10-12 23:47 IJMC: Added 'jointpars1' keyword option.
    # 2014-05-02 11:45 IJMC: Added 'scaleErrors' keyword option..
    # 2014-09-02 19:14 IJMC: Copied from 'phasecurves.py' to 'k2.py'

    params = np.array(arg[0], copy=False)

    if isinstance(arg[-1], dict): 
        # Surreptiously setting keyword arguments:
        kw2 = arg[-1]
        kw.update(kw2)
        arg = arg[0:-1]
    else:
        pass


    if len(arg)==2:
        chisq = errfunc(params, *arg[1], **kw)

    else:
        testfinite = ('testfinite' in kw) and kw['testfinite']
        if not kw.has_key('useindepvar'):
            kw['useindepvar'] = False

        # Keep fixed pairs of joint parameters:
        if kw.has_key('jointpars1'):
            jointpars1 = kw['jointpars1']
            for jointpar1 in jointpars1:
                params[jointpar1[1]] = params[jointpar1[0]]


        if kw.has_key('gaussprior'):
            # If any priors are None, redefine them:
            temp_gaussprior =  kw['gaussprior']
            gaussprior = []
            for pair in temp_gaussprior:
                if pair is None:
                    gaussprior.append([0, np.inf])
                else:
                    gaussprior.append(pair)
        else:
            gaussprior = None

        if kw.has_key('uniformprior'):
            # If any priors are None, redefine them:
            temp_uniformprior =  kw['uniformprior']
            uniformprior = []
            for pair in temp_uniformprior:
                if pair is None:
                    uniformprior.append([-np.inf, np.inf])
                else:
                    uniformprior.append(pair)
        else:
            uniformprior = None

        if kw.has_key('ngaussprior') and kw['ngaussprior'] is not None:
            # If any priors are None, redefine them:
            temp_ngaussprior =  kw['ngaussprior']
            ngaussprior = []
            for triplet in temp_ngaussprior:
                if len(triplet)==3:
                    ngaussprior.append(triplet)
        else:
            ngaussprior = None


        #print "len(arg)>>", len(arg),

        
        if kw.has_key('npars'):
            npars = kw['npars']
            chisq = 0.0
            # Excise "npars" kw for recursive calling:
            lower_kw = kw.copy()
            junk = lower_kw.pop('npars')

            # Keep fixed pairs of joint parameters:
            if kw.has_key('jointpars'):
                jointpars = kw['jointpars']
                for jointpar in jointpars:
                    params[jointpar[1]] = params[jointpar[0]]
                

            for ii in range(len(npars)):
                i0 = sum(npars[0:ii])
                i1 = i0 + npars[ii]
                these_params = arg[0][i0:i1]
                #ret.append(devfunc(these_params, *arg[1][ii], **lower_kw))
                these_params, lower_kw = subfit_kw(arg[0], kw, i0, i1)
                #if 'wrapped_joint_params' in lower_kw:
                #    junk = lower_kw.pop('wrapped_joint_params')
                chisq  += errfunc(these_params, *arg[ii+1], **lower_kw)
                
            return chisq

        else: # Single function-fitting
            depvar = arg[-2]
            weights = arg[-1]

            if not kw['useindepvar']:  # Standard case:
                functions = arg[1]
                helperargs = arg[2:len(arg)-2]
            else:                      # Obsolete, deprecated case:
                functions = arg[1] 
                helperargs = arg[2:len(arg)-3]
                indepvar = arg[-3]



        if testfinite:
            finiteind = isfinite(indepvar) * isfinite(depvar) * isfinite(weights)
            indepvar = indepvar[finiteind]
            depvar = depvar[finiteind]
            weights = weights[finiteind]

        if 'scaleErrors' in kw and kw['scaleErrors']==True:
            if not kw['useindepvar'] or arg[1].__name__=='multifunc' or \
                    arg[1].__name__=='sumfunc':
                model = functions(*((params[1:],)+helperargs))
            else:  # i.e., if useindepvar is True -- old, deprecated usage:
                model = functions(*((params[1:],)+helperargs + (indepvar,)))

            # Compute the weighted residuals:
            if c_chisq:
                chisq = _chi2.chi2(model.ravel(), depvar.ravel(), \
                                       weights.ravel()/params[0]**2)
            else:
                chisq = (weights*((model-depvar)/params[0])**2).sum()
            chisq = chisq/params[0]**2 + 2*depvar.size*np.log(params[0])

        else:
            if not kw['useindepvar'] or arg[1].__name__=='multifunc' or \
                    arg[1].__name__=='sumfunc':
                model = functions(*((params,)+helperargs))
            else:  # i.e., if useindepvar is True -- old, deprecated usage:
                model = functions(*((params,)+helperargs + (indepvar,)))

            # Compute the weighted residuals:
            if c_chisq:
                chisq = _chi2.chi2(model.ravel(), depvar.ravel(), \
                                       weights.ravel())
            else:
                chisq = (weights*(model-depvar)**2).sum()
            

        # Compute 1D and N-D gaussian, and uniform, prior penalties:
        additionalChisq = 0.
        if gaussprior is not None:
            
            additionalChisq += np.sum([((param0 - gprior[0])/gprior[1])**2 for \
                                   param0, gprior in zip(params, gaussprior)])

        if ngaussprior is not None:
            for ind, mu, cov in ngaussprior:
                dvec = params[ind] - mu
                additionalChisq += \
                    np.dot(dvec.transpose(), np.dot(np.linalg.inv(cov), dvec))

        if uniformprior is not None:
            for param0, uprior in zip(params, uniformprior):
                if (param0 < uprior[0]) or (param0 > uprior[1]):
                    chisq *= 1e9

        # Scale up the residuals so as to impose priors in chi-squared
        # space:
        chisq += additionalChisq
    
    return chisq

def xcorrStack(stack, npix_corr=4, plotalot=False, maxshift=np.inf):
    """Return x and y motions determined via 1D cross-correlation of image stack.
    
   :INPUTS:
      stack : 3D numpy array (M x N x N)
        first frame (stack[0]) is the reference image.

      npix_corr : int
         Radius for correlations to measure image motions

    :OUTPUTS:
      dx, dy : tuple of 1D arrays
    """
    # 2011-12-28 19:25 IJMC: Created
    # 2012-06-17 22:06 IJMC: Added plotalot and maxshift flags
    # 2014-09-02 19:21 IJMC: Moved from 'dophot.py' into 'k2.py'

    if plotalot:
        import pylab as py

    nimg, npix, npiy = stack.shape
    sx = stack.mean(1) 
    sy = stack.mean(2) 
    sx0 = sx[0] - sx[0].mean()
    sy0 = sy[0] - sy[0].mean()
    dx = np.zeros(nimg)
    dy = np.zeros(nimg)
    corrCoordx = np.arange(npix) - npix/2
    corrCoordy = np.arange(npiy) - npiy/2
    valid_index = np.abs(corrCoordx) <= maxshift
    valid_indey = np.abs(corrCoordy) <= maxshift
    for ii in range(nimg):
        if np.isfinite(sx[ii]).all() and  np.isfinite(sy[ii]).all():
            corry = np.correlate(sx0, sx[ii]-sx[ii].mean(), mode='same', old_behavior=False)
            corrx = np.correlate(sy0, sy[ii]-sy[ii].mean(), mode='same', old_behavior=False)
            x_guess = corrCoordx[valid_index][(corrx[valid_index]==corrx[valid_index].max())]
            y_guess = corrCoordy[valid_index][(corry[valid_index]==corry[valid_index].max())]
            if np.abs(x_guess) > maxshift:
                x_guess = maxshift * np.sign(x_guess)
            if np.abs(y_guess) > maxshift:
                y_guess = maxshift * np.sign(y_guess)

            x_ind = np.abs(corrCoordx - x_guess) < npix_corr
            y_ind = np.abs(corrCoordy - y_guess) < npix_corr
            xfit = np.polyfit(corrCoordx[x_ind], corrx[x_ind], 2)
            yfit = np.polyfit(corrCoordy[y_ind], corry[y_ind], 2)
            dx[ii] = -0.5 * xfit[1] / xfit[0]
            dy[ii] = -0.5 * yfit[1] / yfit[0]
            if np.abs(dx[ii]) > maxshift:
                dx[ii] = maxshift * np.sign(dx[ii])
            if np.abs(dy[ii]) > maxshift:
                dy[ii] = maxshift * np.sign(dy[ii])

            if plotalot:
                py.figure()
                py.plot(corrCoordx[x_ind], corrx[x_ind], 'ob', label='x')
                py.plot(corrCoordy[y_ind], corry[y_ind], 'dr', label='y')
                py.plot(corrCoordx[x_ind], np.polyval(xfit, corrCoordx[x_ind]), '--b')
                py.plot(corrCoordy[y_ind], np.polyval(yfit, corrCoordy[y_ind]), '--r')
        else:
            dx[ii] = 0
            dy[ii] = 0

    return dx, dy


if __name__ == "__main__":
    sys.exit(main())


