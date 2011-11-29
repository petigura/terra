"""
Functions for facilitating the reading and writing of Kepler files.
"""


import atpy
import os
import glob
import pyfits

kepdir = os.environ['KEPDIR']

def KICPath(KIC,basedir):
    """
    KIC     - Target star identifier.
    basedir - Directory leading to files.
              'orig' - prestine kepler data
              'clip' - clipped data
              'dt'   - detrended data.
    """

    if basedir is 'orig':
        basedir = 'kepdat/EX/Q*/'
    if basedir is 'clip':
        basedir = 'tempfits/clip/'
    if basedir is 'dt':
        basedir = 'kepdat/DT/'

    basedir = os.path.join(kepdir,basedir)
    g = glob.glob(basedir+'*%09i*.fits' % KIC)
    return g

def mqload(files):
    """
    Load up a table set given a list of fits files.
    """
    tset = atpy.TableSet()
    
    for f in files:
        hdu = pyfits.open(f)
        t = atpy.Table(f,type='fits')
        t.table_name = 'Q%i' % hdu[0].header['QUARTER']
        t.add_keyword('PATH',f)
        tset.append(t)

    return tset 
