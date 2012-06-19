"""
Make diagnostic plots for the mode fitting
"""

from argparse import ArgumentParser
import h5py
from matplotlib.pylab import *
import os
import atpy

kicdbpath = os.environ['KEPBASE']+'files/KIC.db'

parser = ArgumentParser(description='Diagnose Modes')
parser.add_argument('lc',  type=str,help='Input LC file')
parser.add_argument('dt',  type=str,help='Detrended Flux')
parser.add_argument('svd',  type=str,help='Mode file')
parser.add_argument('out',  type=str,help='outputbase name')

args = parser.parse_args()
nModes = 4
hlc = h5py.File(args.lc)
hdt = h5py.File(args.dt)
hsvd = h5py.File(args.svd)

fdt = hdt['LIGHTCURVE']['fdt']
U,S,V,goodid,mad = hsvd['U'],hsvd['S'],hsvd['V'],hsvd['goodid'],hsvd['MAD']

nstars = U.shape[0]
S = S[:nstars,:nstars]
A = dot(U,S)
fit = dot(A[:,:nModes],V[:nModes])
fit = fit*mad[goodid]

kic = hlc['KIC']
a = tuple(kic[:])
query = 'select id,kic_ra,kic_dec from KIC where id in %s' % str(a)

#import pdb
#pdb.set_trace()
tkic = atpy.Table('sqlite',kicdbpath,table='KIC',query=query)
tkic = tkic.rows(goodid)
scatter(tkic.kic_ra,tkic.kic_dec)

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

A = A[:,:nModes]
cmax = abs(A).max()

for i in range(nModes):
    ra = tkic.kic_ra
    dec = tkic.kic_dec
    scatter(ra,dec,c=A[:,i],edgecolors='none',vmin=-cmax,vmax=cmax,s=60)
    xlabel('RA (Deg)')
    ylabel('Dec (Deg)')
    fig = gcf()
    fig.savefig(args.out+'_fov%i.png' % i)

