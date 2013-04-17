#!/usr/bin/env python

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

dbdir = os.environ['KEPBASE']+'files/db/'
qdbpath = dbdir+'quarters.db'
kicdbpath = dbdir+'kic_ct.db'

nModes = 4

parser = ArgumentParser(description='Diagnose Modes')
parser.add_argument('svd',type=str,help='Mode file')
parser.add_argument('out',nargs='?',type=str,help='output basename.  If none is given, just knock off the .svd.h5')

args = parser.parse_args()
out  = args.out 
if out is None:
    out = args.svd.replace('.svd.h5','')

hsvd = h5py.File(args.svd)

V   = hsvd['V'][:]
kic = hsvd['KIC'][:]
A   = hsvd['A'][:]


# Sort the A array in ascending order of KIC 
gkic = kic[:]
sid  = argsort(gkic)
gkic = gkic[sid]
A    =  A[sid]

skic = str(tuple(gkic))

# Pull the RA and Dec information from the sqlite3 database.
conn = sqlite3.connect(kicdbpath)
cur = conn.cursor()

query = """
SELECT 
kic,kic_ra,kic_dec FROM kic WHERE kic in %(skic)s;
""" % {'skic':skic} 
cur.execute(query)
res = cur.fetchall()
conn.close()

# Verify that the SQL KIC IDs match the ones we searched for.
tkic = array(res,dtype=zip(['skic','ra','dec'],[int,float,float]))
assert (tkic['skic'] == gkic).all(),'h5 KIC does not match SQL KIC'

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
    fig.savefig(out+'_fov%i.png' % i)
clf()

t = np.arange(V[0].size)*keptoy.lc
for i in range(nModes):
    step = np.std(V[i])
    plot(t,V[i]+i*step*5)

xlabel('time')
title('%i Modes' % nModes)
fig = gcf()
fig.savefig(out+'_pc.png')


