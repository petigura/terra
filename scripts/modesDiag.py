"""
Make diagnostic plots for the mode fitting
"""

from argparse import ArgumentParser
import h5py
from matplotlib.pylab import *
import os
import atpy
import sqlite3

kicdbpath = os.environ['KEPBASE']+'files/KIC.db'
nModes = 4
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

plt.register_cmap(name='BlueRed3', data=cdict3) # optional lut kwarg
plt.rcParams['image.cmap'] = 'BlueRed3'

parser = ArgumentParser(description='Diagnose Modes')
parser.add_argument('dt', type=str,help='Detrended Flux')
parser.add_argument('svd',type=str,help='Mode file')
parser.add_argument('q',  type=int,help='quarter')
parser.add_argument('out',type=str,help='outputbase name')

args = parser.parse_args()
hdt = h5py.File(args.dt)
hsvd = h5py.File(args.svd)
q    = args.q

fdt = hdt['LIGHTCURVE']['fdt']
t   = hdt['LIGHTCURVE1d']['t']

U,S,V,goodid,mad,kic = \
    hsvd['U'],hsvd['S'],hsvd['V'],hsvd['goodid'],hsvd['MAD'],hsvd['KIC']

# Construct fits
nstars = U.shape[0]
S     = S[:nstars,:nstars]
A     = dot(U,S)
fit   = dot(A[:,:nModes],V[:nModes])
fit   = fit*mad[goodid]

skic = str(tuple(kic[goodid]))

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

tkic = array(res,dtype=zip(['id','ra','dec'],[int,float,float]))
A = A[:,:nModes]
cmax = abs(A).max()
for i in range(nModes):
    ra = tkic['ra']
    dec = tkic['dec']
    scatter(ra,dec,c=A[:,i],edgecolors='none',vmin=-cmax,vmax=cmax,s=30)
    xlabel('RA (Deg)')
    ylabel('Dec (Deg)')
    title('PC %i' % (i+1))
    fig = gcf()
    fig.savefig(args.out+'_fov%i.png' % i)

clf()
for i in range(nModes):
    step = np.std(V[i])
    plot(t,V[i]+i*step*5)

xlabel('time')
title('%i Modes' % nModes)
fig = gcf()
fig.savefig(args.out+'_pc.png')


