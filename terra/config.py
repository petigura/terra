import numpy as np
import os

# Setup directories
#k2_dir = os.environ['K2_DIR']
#k2_camp = os.environ['K2_CAMP']

#def resolve_grid(outfile):
#    """
#    Determine path to .grid.h5 based on environment variables
#    """
#    if os.environ['K2_SEARCH_DIR']=='':
#        print "K2_SEACH_DIR not set. Using K2_DIR"
#        k2_projdir = os.environ['K2_DIR']
#    else:
#        k2_projdir = os.environ['K2_SEARCH_DIR']
#
#    if type=='grid':
#        return "%s/%s" % (k2_projdir,outfile)
#
#svdh5 = "%s/Ceng.svd.h5" % (k2_dir) # h5 structure to read mode info from
#
# path to pixel data
#path_pix_fits = os.environ['K2_PIX_DIR']

# path to light curve photometry
#path_phot = "%s/%s.phot.h5" % (os.environ['K2_PHOT_DIR'],k2_camp) 
#path_train = "%s/%s.train.h5" % (os.environ['K2_PHOT_DIR'],k2_camp) 


# keptoy
sc = 58.8488 /60./60./24. # SC time in days
lc = sc*30 # LC exp time in days
P05c = 100   # Parameter that controls how sharp P05 edge is.

G          = 6.672e-8 # [cm3 g^-1 s^-2]
Rsun       = 6.955e10 # cm
Rearth     = 6.378e8  # cm
Msun       = 1.989e33 # [g]
AU         = 1.496e13 # [cm]
sec_in_day = 86400.
stefan_boltzmann = 5.6704e-5 # [erg cm^-2 s^-1 K^-4]

# sim
blockSize = 1000

# tfind
miTransMES = 3 # Minimum number of transits to even be included in MES
miSES      = 0 # All transits must have SES larger than this.
twdG       = [3,5,7,10,14,18,23,28,32,38]
P1,P2      = 5,50 # Days over which to compute periodogram 
maCadCnt   = 5e3 # cadCount counts up the number of times a particular
                 # cadence goes into the MES calculation.  If a
                 # certain cadence is drastically over represented,
                 # clip it.

# tval
dP     = 0.5
depoch = 0.5
hs2n   = 5   # Search for harmonics if signal has S/N larger than this.

# Try the following harmonics in harm.
from fractions import Fraction
harmL = np.array([1,Fraction('1/2'),Fraction('1/3'),2,3])
nCheck = 10 # Number of peaks in periodogram to fit.

#modesDiag
cdict3 = {'red':  ((0.0, 0.0, 0.0),
                   (0.25,0.0, 0.0),
                   (0.5, 0.8, 1.0),
                   (0.75,1.0, 1.0),
                   (1.0, 0.4, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.25,0.0, 0.0),
                   (0.5, 0.9, 0.9),
                   (0.75,0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.4),
                   (0.25,1.0, 1.0),
                   (0.5, 1.0, 0.8),
                   (0.75,0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }


# Cotrending
nMode     = 8
nModeSave = 20 # To save space, don't save all the of the V matrix
sigOut    = 10
maxIt     = 10 # Maximum number of iterations for robust-SVD

# Prepro
stepThrsh = 5e-3 # Threshold steps must be larger than.  
wd        = 4    # Width of the median filter used to determine step

# After identifying problem cadences, we expand the region which we
# consider bad data by the following amounts in both directions.
cadGrow = dict(desat=1,atTwk=4)

climb_solar = np.array([ 0.77, -0.67, 1.14, -0.41])
