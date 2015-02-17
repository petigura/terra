import argparse
import os
import h5py
import prepro


desc = """
Wrapper around prepro
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('lc', type=str, help='light curve *.h5')
parser.add_argument('cmds',nargs='+',type=str,help='dt, cal, mqcal')
parser.add_argument('--svd_folder', type=str,help='folder containing SVD matrices')
parser.add_argument('--fits',nargs='+',type=str,help='list of input fits files')
parser.add_argument('--fields',nargs='+',type=str,help='keep fields')
args = parser.parse_args()

with h5py.File(args.lc) as h5:
    if args.fields==None:
        fields=[]
    else:
        fields=args.fields


    for cmd in args.cmds:
        if cmd=='cal':
            prepro.cal(h5,args.svd_folder)
        elif cmd=='raw':
            prepro.raw(h5,args.fits,fields=fields)


        print "%s: %s" % (h5,cmd)
