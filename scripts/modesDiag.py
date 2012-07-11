"""
Make diagnostic plots for the mode fitting

Future Work
-----------
- Figure out a better way to attach Kepler ID to the Rows rows of the h5 table.
- Move plotting to a different function, so I can call it interactively.

"""

from argparse import ArgumentParser
import h5py
import numpy as np
from matplotlib.pylab import *
import os
import sqlite3

import keptoy

# Import and update color scale.
from config import cdict3
plt.register_cmap(name='BlueRed3', data=cdict3) # optional lut kwarg
plt.rcParams['image.cmap'] = 'BlueRed3'

kicdbpath = os.environ['KEPBASE']+'files/KIC.db'
nModes = 4

parser = ArgumentParser(description='Diagnose Modes')
parser.add_argument('svd',type=str,help='Mode file')
parser.add_argument('q',  type=int,help='quarter')
parser.add_argument('out',type=str,help='output basename')
args = parser.parse_args()

hsvd = h5py.File(args.svd)
q    = args.q

U,S,V,goodid,mad,kic = \
    hsvd['U'],hsvd['S'],hsvd['V'],hsvd['goodid'],hsvd['MAD'],hsvd['KIC']

# Construct fits
nstars = U.shape[0]
S     = S[:nstars,:nstars]
A     = dot(U,S)
A     = A[:,:nModes]
fit   = dot(A,V[:nModes])
fit   = fit*mad[goodid]

# Order the KIC list and the Fit Coeffs by KIC
gkic = kic[goodid]
sid  = argsort(gkic)
gkic = gkic[sid]
A = A[sid]



skic = str(tuple(gkic))

# Pull the RA and Dec information from the sqlite3 database.
conn = sqlite3.connect(kicdbpath)
cur = conn.cursor()
query = """
SELECT 
q%(q)i.id,kic.kic_ra,kic.kic_dec 
FROM q%(q)i 
JOIN kic 
ON q%(q)i.id=kic.id 
WHERE q%(q)i.id in %(skic)s
""" % {'q':q,'skic':skic} 
cur.execute(query)
res = cur.fetchall()
conn.close()

# Verify that the SQL KIC IDs match the ones we searched for.
tkic = array(res,dtype=zip(['id','ra','dec'],[int,float,float]))
assert (tkic['id'] == gkic).all(),'h5 KIC does not match SQL KIC'

# Plot distribtion of mode weights across the FOV.
cmax = abs(A).max()
fig = figure(figsize=(18,8))
for i in range(nModes):
    scatter(tkic['ra'],tkic['dec'],c=A[:,i],edgecolors='none',vmin=-cmax,
            vmax=cmax,s=30)
    xlabel('RA (Deg)')
    ylabel('Dec (Deg)')
    title('PC %i' % (i+1))
    fig = gcf()
    fig.savefig(args.out+'_fov%i.png' % i)
clf()

t = np.arange(V[0].size)*keptoy.lc
for i in range(nModes):
    step = np.std(V[i])
    plot(t,V[i]+i*step*5)

xlabel('time')
title('%i Modes' % nModes)
fig = gcf()
fig.savefig(args.out+'_pc.png')


