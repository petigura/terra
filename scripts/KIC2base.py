import sim
import argparse
import atpy
import os
import numpy as np
import keplerio
import glob

parser = argparse.ArgumentParser(
    description='KIC')

parser.add_argument('DataDir',type=str)
parser.add_argument('OutDir',type=str)
parser.add_argument('KIC',type=int)

args = parser.parse_args()

KIC = args.KIC
# Find files with KIC name
path = os.path.join(
    args.DataDir,
    'archive/data3/privkep/EX/Q?/kplr%09d-*_llc.fits' % KIC)
files = glob.glob(path)
tBASE = map(keplerio.qload,files)
tBASE = map(keplerio.nQ,tBASE)
tBASE = atpy.TableSet(tBASE)

basepath = os.path.join(args.OutDir,'KIC_%09d.fits' % KIC)
tBASE.write(basepath,overwrite=True)

