import sim
import argparse
import atpy
import os
import numpy as np
import keplerio

parser = argparse.ArgumentParser(
    description='Read in basefile inject a transit')

parser.add_argument('PARfile',type=str)
args = parser.parse_args()

tPAR = atpy.Table(args.PARfile,type='fits')
KICL = np.unique(tPAR.KIC)
dir = os.path.dirname(args.PARfile)

for KIC in KICL:
    basepath = os.path.join(dir,'tBASE_%09d.fits' % KIC)
#    if not os.path.exists(basepath):
    files = keplerio.KICPath(KIC,'orig')
    tBASE = map(keplerio.qload,files)
    tBASE = map(keplerio.nQ,tBASE)
    tBASE = atpy.TableSet(tBASE)
    tBASE.write(basepath,overwrite=True)

