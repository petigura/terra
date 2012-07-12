import numpy as np

# Qalg
Plim =  0.001   # Periods must agree to this fractional amount
epochlim =  0.1 # Epochs must agree to 0.1 days   

# keptoy
G = 6.67e-8 # cgs
R_sun = 6.9e10 # cm
M_sun = 2.0e33 # g
lc = 0.0204343960431288
c = 100   # Parameter that controls how sharp P05 edge is.

# sim
blockSize = 1000

# tfind
miTransMES = 3 # Minimum number of transits to even be included in MES
miSES      = 0 # All transits must have SES larger than this.

# tval
dP     = 0.5
depoch = 0.5
hs2n   = 5   # Search for harmonics if signal has S/N larger than this.

# Try the following harmonics in harm.
harmL = np.array( [1/2., 2/3., 2., 3/2., 4/3. ] )
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
