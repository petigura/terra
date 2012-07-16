from argparse import ArgumentParser
from numpy import ma
import keptoy
import h5py
import h5plus
import numpy as np
import tfind
import cotrend

prsr = ArgumentParser(description='Run grid search')

prsr.add_argument('inp',  type=str   , help='input file')
prsr.add_argument('out',  type=str   , help='output file')
prsr.add_argument('--kic',type=int,nargs='+',help= 'KIC IDs to process')
args = prsr.parse_args()

h5     = h5py.File(args.inp)
dslc   = h5['LIGHTCURVE']
dskic  = h5['KIC'][:]

# Select a subset of the stars.
if args.kic is not None:
    argkic = np.array(args.kic)
    x2 = x1 = np.arange( dslc.shape[0] )

    idargj,iddsj,kic = cotrend.join_on_kic(x1,x2,argkic,dskic)
    iddsj = list(iddsj)
    
    skic = kic.astype('|S10')
    skic = reduce(lambda x,y : x+' '+y,skic)
    dslc = dslc[iddsj]

    assert kic.size==argkic.size,'kic star not found'

    print "Running grid search on the following %i stars \n" % kic.size
    print skic

nstars,ncad = dslc.shape

rL = []
for i in range(nstars):    
    lc = dslc[i]
    t = lc['t']
    fcal = ma.masked_array(lc['fcal'],lc['fmask'],fill_value=0)

    isStep = np.zeros(fcal.size).astype(bool)

    P1 = int(np.floor(5./keptoy.lc))
    P2 = int(np.floor(50./keptoy.lc))
    twdG = [3,5,7,10,14,18]

    rtd = tfind.tdpep(t,fcal,isStep,P1,P2,twdG)
    r   = tfind.tdmarg(rtd)
    rL.append(r)

rL = np.vstack(rL)

h5 = h5plus.File(args.out)
h5.create_dataset('RES',data=rL)
h5.create_dataset('KIC',data=kic)
h5.close()
print "grid: Created %s" % args.out
