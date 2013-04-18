import numpy as np

# Qalg
Plim =  0.001   # Periods must agree to this fractional amount
epochlim =  0.1 # Epochs must agree to 0.1 days   

# keptoy
G = 6.67e-8 # cgs
R_sun = 6.9e10 # cm
M_sun = 2.0e33 # g
sc = 58.8488 /60./60./24. # SC time in days
lc = sc*30 # LC exp time in days
c = 100   # Parameter that controls how sharp P05 edge is.

# sim
blockSize = 1000

# tfind
miTransMES = 3 # Minimum number of transits to even be included in MES
miSES      = 0 # All transits must have SES larger than this.
twdG       = [3,5,7,10,14,18]
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
nMode     = 4
nModeSave = 20 # To save space, don't save all the of the V matrix
sigOut    = 10
maxIt     = 10 # Maximum number of iterations for robust-SVD

# Prepro
stepThrsh = 5e-3 # Threshold steps must be larger than.  
wd        = 4    # Width of the median filter used to determine step

# After identifying problem cadences, we expand the region which we
# consider bad data by the following amounts in both directions.
cadGrow = dict(desat=1,atTwk=4)


